import re
from collections import Counter
from statistics import mean
import pandas as pd

# 标点符号正则
_PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~„“”’…«»"""
PUNCT_RE = re.compile(rf"^[{re.escape(_PUNCT)}]+|[{re.escape(_PUNCT)}]+$")

def tokenize(line: str):
    toks = []
    for t in line.strip().split():
        t = PUNCT_RE.sub("", t)
        if t:
            toks.append(t)
    return toks

def file_stats(path: str, encoding="utf-8"):
    n_sent = 0
    sent_lens = []
    token_counter = Counter()

    with open(path, "r", encoding=encoding, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_sent += 1
            toks = tokenize(line)
            sent_lens.append(len(toks))
            token_counter.update(toks)

    n_tokens = sum(token_counter.values())
    n_types = len({t.lower() for t in token_counter.keys()})
    avg_len = mean(sent_lens) if sent_lens else 0.0
    min_len = min(sent_lens) if sent_lens else 0
    max_len = max(sent_lens) if sent_lens else 0
    return {
        "Sentences": n_sent,
        "Tokens": n_tokens,
        "Types": n_types,
        "Avg Len": round(avg_len, 1),
        "Min Len": min_len,
        "Max Len": max_len
    }

def compare_corpora(german_path, sorbian_path):
    de_stats = file_stats(german_path)
    hsb_stats = file_stats(sorbian_path)
    return pd.DataFrame([de_stats, hsb_stats], index=["German (de)", "Upper Sorbian (hsb)"])


if __name__ == "__main__":
    SORBIAN_PATH = "/content/drive/MyDrive/Colab Notebooks/train.hsb-de.hsb"
    GERMAN_PATH  = "/content/drive/MyDrive/Colab Notebooks/train.hsb-de.de"

    df = compare_corpora(GERMAN_PATH, SORBIAN_PATH)
    print(df)
