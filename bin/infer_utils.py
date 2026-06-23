import re
from collections import defaultdict, Counter

SPECIAL_TOKENS = ["<ts>", "<te>", "<bc>", "<silence>", "<pause>"]
SPEAKERS = ["speaker_A", "speaker_B"]

TOKEN_PATTERN = re.compile(
    r"(<speaker_A>|</speaker_A>|<speaker_B>|</speaker_B>|<ts>|<te>|<bc>|<silence>|<pause>)"
)


def parse_sequence(s: str):
    """
    输出:
    [
      ("text", "speaker_A", "ええ、そう..."),
      ("special", None, "<silence>"),
      ("text", "speaker_B", "はい"),
      ...
    ]
    """
    parts = TOKEN_PATTERN.split(s)
    events = []
    cur_speaker = None

    for p in parts:
        if not p:
            continue

        if p == "<speaker_A>":
            cur_speaker = "speaker_A"
        elif p == "</speaker_A>":
            cur_speaker = None
        elif p == "<speaker_B>":
            cur_speaker = "speaker_B"
        elif p == "</speaker_B>":
            cur_speaker = None
        elif p in SPECIAL_TOKENS:
            events.append(("special", None, p))
        else:
            text = p.strip()
            if text:
                events.append(("text", cur_speaker, text))

    return events

def levenshtein(a, b):
    """
    a, b 可以是 list[str]，这里用于 char-level CER
    """
    n, m = len(a), len(b)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = tmp

    return dp[m]


def collect_speaker_text(seq: str):
    events = parse_sequence(seq)
    speaker_text = defaultdict(list)

    for typ, spk, value in events:
        if typ == "text" and spk in SPEAKERS:
            speaker_text[spk].append(value)

    return {
        spk: "".join(texts)
        for spk, texts in speaker_text.items()
    }


def speaker_cer(pred: str, ref: str):
    pred_text = collect_speaker_text(pred)
    ref_text = collect_speaker_text(ref)

    results = {}
    total_edits = 0
    total_ref_chars = 0

    for spk in SPEAKERS:
        hyp = pred_text.get(spk, "")
        tgt = ref_text.get(spk, "")

        dist = levenshtein(list(hyp), list(tgt))
        denom = max(len(tgt), 1)
        cer = dist / denom

        results[spk] = {
            "hyp": hyp,
            "ref": tgt,
            "edits": dist,
            "ref_chars": len(tgt),
            "cer": cer,
        }

        total_edits += dist
        total_ref_chars += len(tgt)

    results["micro_avg"] = {
        "edits": total_edits,
        "ref_chars": total_ref_chars,
        "cer": total_edits / max(total_ref_chars, 1),
    }

    return results

def lcs_match_count(a, b):
    """
    返回两个 token 序列的最长公共子序列长度
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[n][m]

def collect_special_tokens(seq: str):
    events = parse_sequence(seq)
    return [value for typ, _, value in events if typ == "special"]

def special_token_f1_sequence(pred: str, ref: str):
    pred_tokens = collect_special_tokens(pred)
    ref_tokens = collect_special_tokens(ref)

    tp = lcs_match_count(pred_tokens, ref_tokens)
    fp = len(pred_tokens) - tp
    fn = len(ref_tokens) - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "pred_tokens": pred_tokens,
        "ref_tokens": ref_tokens,
        "tp_lcs": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
if __name__ == "__main__":
    ref = "language Japanese<asr_text><speaker_B>それさえもなんかいやったいいやったい食べるので</speaker_B><ts><speaker_A>あっ</speaker_A><pause><speaker_A>そうなんだ</speaker_A>"
    pred = "<speaker_B>それさえもなんかや</speaker_B><bc><speaker_A>ん</speaker_A><bc><speaker_B>ってなると食べるの（笑language Japanese<asr_text><speaker_B>それさえも、なんか、やったいやったり食べるので</speaker_B><ts><speaker_A>あ</speaker_A><pause><speaker_A>そうなんだ</speaker_A>"
    