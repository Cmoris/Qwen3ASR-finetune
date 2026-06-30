import argparse
import glob
import json
from pathlib import Path

from infer_utils import SPECIAL_TOKENS, special_token_classification_metrics


def new_summary_item():
    return {
        "num_samples": 0,
        "speaker": {
            "speaker_A": {"edits": 0, "ref_chars": 0},
            "speaker_B": {"edits": 0, "ref_chars": 0},
            "micro_avg": {"edits": 0, "ref_chars": 0},
        },
        "special_sequence": {"tp": 0, "fp": 0, "fn": 0},
        "special_token_classification": {
            "per_token": {
                token: {"tp": 0, "fp": 0, "fn": 0}
                for token in SPECIAL_TOKENS
            },
        },
    }


def update_summary(item, metrics):
    item["num_samples"] += 1

    speaker_cer = metrics["speaker_cer"]
    for speaker in ("speaker_A", "speaker_B", "micro_avg"):
        source = speaker_cer[speaker]
        target = item["speaker"][speaker]
        target["edits"] += int(source.get("edits", 0))
        target["ref_chars"] += int(source.get("ref_chars", 0))

    sequence = metrics["special_token_f1_sequence"]
    item["special_sequence"]["tp"] += int(
        sequence.get("tp_lcs", sequence.get("tp", 0))
    )
    item["special_sequence"]["fp"] += int(sequence.get("fp", 0))
    item["special_sequence"]["fn"] += int(sequence.get("fn", 0))

    per_token = metrics["special_token_classification"]["per_token"]
    for token in SPECIAL_TOKENS:
        source = per_token[token]
        target = item["special_token_classification"]["per_token"][token]
        for field in ("tp", "fp", "fn"):
            target[field] += int(source.get(field, 0))


def precision_recall_f1(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def finalize_summary(item):
    for stat in item["speaker"].values():
        stat["cer"] = stat["edits"] / max(stat["ref_chars"], 1)

    sequence = item["special_sequence"]
    precision, recall, f1 = precision_recall_f1(
        sequence["tp"], sequence["fp"], sequence["fn"]
    )
    sequence.update(precision=precision, recall=recall, f1=f1)

    classification = item["special_token_classification"]
    per_token = classification["per_token"]
    for stat in per_token.values():
        stat["predicted"] = stat["tp"] + stat["fp"]
        stat["support"] = stat["tp"] + stat["fn"]
        precision, recall, f1 = precision_recall_f1(
            stat["tp"], stat["fp"], stat["fn"]
        )
        stat.update(precision=precision, recall=recall, f1=f1)

    total_tp = sum(stat["tp"] for stat in per_token.values())
    total_fp = sum(stat["fp"] for stat in per_token.values())
    total_fn = sum(stat["fn"] for stat in per_token.values())
    total_support = total_tp + total_fn
    precision, recall, f1 = precision_recall_f1(
        total_tp, total_fp, total_fn
    )
    classification["micro_avg"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted": total_tp + total_fp,
        "support": total_support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    classification["macro_avg"] = {
        metric: sum(stat[metric] for stat in per_token.values())
        / max(len(per_token), 1)
        for metric in ("precision", "recall", "f1")
    }
    classification["macro_avg"]["support"] = total_support

    classification["weighted_avg"] = {
        metric: sum(
            stat[metric] * stat["support"] for stat in per_token.values()
        )
        / max(total_support, 1)
        for metric in ("precision", "recall", "f1")
    }
    classification["weighted_avg"]["support"] = total_support

    return {"overall": item}


def sampler_identity(row):
    """Fields stable across duplicate DistributedSampler evaluations."""
    return (
        str(row.get("conv_ids", "")),
        row.get("ref_text", ""),
        row.get("pred_text", ""),
    )


def remove_sampler_padding(shards):
    """
    Remove only trailing rows that duplicate a row at the head of a shard.

    torch DistributedSampler pads with at most world_size - 1 samples copied
    from the start of the dataset. Those copies occur at shard tails.
    """
    if len(shards) <= 1:
        return 0

    head_identities = {
        sampler_identity(row)
        for rows in shards
        for row in rows[:1]
    }
    removed = 0
    max_padding = len(shards) - 1

    for rows in reversed(shards):
        if removed >= max_padding:
            break
        if rows and sampler_identity(rows[-1]) in head_identities:
            rows.pop()
            removed += 1

    return removed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge sharded Qwen3-ASR non-streaming evaluation results."
    )
    parser.add_argument(
        "--input_glob",
        required=True,
        help="Glob matching rank JSONL files; quote it in the shell.",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument(
        "--keep_sampler_padding",
        action="store_true",
        help="Keep duplicate tail rows introduced by DistributedSampler.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_resolved = output_path.resolve()
    paths = [
        Path(path)
        for path in sorted(glob.glob(args.input_glob))
        if Path(path).resolve() != output_resolved
    ]
    if not paths:
        raise RuntimeError(f"No input files matched: {args.input_glob}")

    shards = []
    for path in paths:
        rows = []
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                rows.append(row)
        shards.append(rows)

    removed = 0
    if not args.keep_sampler_padding:
        removed = remove_sampler_padding(shards)

    indexed_rows = []
    for shard_index, rows in enumerate(shards):
        for row_index, row in enumerate(rows):
            local_index = int(row.get("sample_idx", row_index))
            indexed_rows.append(
                (local_index, shard_index, row_index, row)
            )
    indexed_rows.sort(key=lambda value: value[:3])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_item = new_summary_item()
    with output_path.open("w", encoding="utf-8") as destination:
        for merged_index, (_, _, _, row) in enumerate(indexed_rows):
            metrics = row.setdefault("metrics", {})
            if "special_token_classification" not in metrics:
                if "pred_text" not in row or "ref_text" not in row:
                    raise KeyError(
                        "Cannot compute special-token metrics: a row is "
                        "missing pred_text or ref_text"
                    )
                metrics["special_token_classification"] = (
                    special_token_classification_metrics(
                        pred=row["pred_text"],
                        ref=row["ref_text"],
                    )
                )

            # rank-local sample_idx values overlap, so replace them with a
            # globally unique index in merged order.
            row["sample_idx"] = merged_index
            destination.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            update_summary(summary_item, metrics)

    summary = finalize_summary(summary_item)
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)

    print(f"Merged {len(paths)} files and {len(indexed_rows)} samples")
    if removed:
        print(f"Removed {removed} DistributedSampler padding sample(s)")
    print(f"Saved merged results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
