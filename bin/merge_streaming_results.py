import argparse
import json
from pathlib import Path

from infer_utils import SPECIAL_TOKENS, special_token_classification_metrics


def empty_summary():
    return {}


def update_summary(summary, step_ms: int, metrics: dict):
    step_key = str(step_ms)

    if step_key not in summary:
        summary[step_key] = {
            "num_samples": 0,
            "speaker": {
                "speaker_A": {"edits": 0, "ref_chars": 0},
                "speaker_B": {"edits": 0, "ref_chars": 0},
                "micro_avg": {"edits": 0, "ref_chars": 0},
            },
            "special_sequence": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
            },
            "special_token_classification": {
                "per_token": {
                    token: {"tp": 0, "fp": 0, "fn": 0}
                    for token in SPECIAL_TOKENS
                },
            },
        }

    item = summary[step_key]
    item["num_samples"] += 1

    cer = metrics["speaker_cer"]

    for spk in ["speaker_A", "speaker_B", "micro_avg"]:
        if spk in cer:
            item["speaker"][spk]["edits"] += int(cer[spk].get("edits", 0))
            item["speaker"][spk]["ref_chars"] += int(cer[spk].get("ref_chars", 0))

    f1 = metrics["special_token_f1_sequence"]
    tp = f1.get("tp_lcs", f1.get("tp", 0))
    fp = f1.get("fp", 0)
    fn = f1.get("fn", 0)

    item["special_sequence"]["tp"] += int(tp)
    item["special_sequence"]["fp"] += int(fp)
    item["special_sequence"]["fn"] += int(fn)

    classification = metrics["special_token_classification"]
    for token in SPECIAL_TOKENS:
        source = classification["per_token"][token]
        target = item["special_token_classification"]["per_token"][token]
        for field in ("tp", "fp", "fn"):
            target[field] += int(source.get(field, 0))


def finalize_summary(summary):
    for step_key, item in summary.items():
        for spk, stat in item["speaker"].items():
            edits = stat["edits"]
            ref_chars = stat["ref_chars"]
            stat["cer"] = edits / max(ref_chars, 1)

        s = item["special_sequence"]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        s["precision"] = precision
        s["recall"] = recall
        s["f1"] = f1

        classification = item["special_token_classification"]
        per_token = classification["per_token"]
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_support = 0

        for stat in per_token.values():
            tp, fp, fn = stat["tp"], stat["fp"], stat["fn"]
            stat["predicted"] = tp + fp
            stat["support"] = tp + fn
            stat["precision"] = tp / max(tp + fp, 1)
            stat["recall"] = tp / max(tp + fn, 1)
            stat["f1"] = (
                2
                * stat["precision"]
                * stat["recall"]
                / max(stat["precision"] + stat["recall"], 1e-8)
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_support += stat["support"]

        micro_precision = total_tp / max(total_tp + total_fp, 1)
        micro_recall = total_tp / max(total_tp + total_fn, 1)
        classification["micro_avg"] = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "predicted": total_tp + total_fp,
            "support": total_support,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": (
                2
                * micro_precision
                * micro_recall
                / max(micro_precision + micro_recall, 1e-8)
            ),
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

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.input_glob))
    if len(paths) == 0:
        raise RuntimeError(f"No files matched: {args.input_glob}")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = empty_summary()

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)
                rows.append(obj)

    rows.sort(key=lambda x: (x.get("sample_idx", 0), x.get("step_ms", 0)))

    with output_path.open("w", encoding="utf-8") as fout:
        for obj in rows:
            metrics = obj.setdefault("metrics", {})
            if "special_token_classification" not in metrics:
                if "pred_text" not in obj or "ref_text" not in obj:
                    raise KeyError(
                        "Cannot compute special-token classification metrics: "
                        "row is missing pred_text or ref_text"
                    )
                metrics["special_token_classification"] = (
                    special_token_classification_metrics(
                        pred=obj["pred_text"],
                        ref=obj["ref_text"],
                    )
                )

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            update_summary(summary, obj["step_ms"], metrics)

    summary = finalize_summary(summary)

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(paths)} files into {output_path}")
    print(f"Saved merged summary to {summary_path}")


if __name__ == "__main__":
    main()
