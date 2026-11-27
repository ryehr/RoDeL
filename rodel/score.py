"""Score the predictions written by evaluate.py.

A point is <predictions tag>.<configuration>, e.g. seq_r3.ks. For every point the script prints the F1 of
each test set (mean over its questions), the plain average over the sets, and a bootstrap confidence
interval of that average. --pairs prints paired differences between two points.

Scores: token F1 against the closest reference, as stored by evaluate.py. LoCoMo uses the scoring of its
benchmark (Porter-stemmed token F1 against the first reference; partial credit over comma-separated
sub-answers for category-1 questions; the category is read from the question's "extra" field).
Bootstrap: sets are resampled independently, by document (by question for LoCoMo), with the same draws
for both points of a pair.

  python rodel/score.py --run-dir runs/seq --points seq_r3.k seq_r3.ks --pairs seq_r3.ks-seq_r3.k \
      --sets hotpotqa musique --label in-domain
"""
import argparse
import glob
import os
import random
import re
import string
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dataset_of, read_jsonl

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--points", nargs="*", default=[])
ap.add_argument("--pairs", nargs="*", default=[], help="x-y: difference between two points")
ap.add_argument("--label", default="average")
ap.add_argument("--B", type=int, default=2000, help="bootstrap resamples")
a = ap.parse_args()

_stem = None


def stem(w):
    global _stem
    if _stem is None:
        from nltk.stem import PorterStemmer
        _stem = PorterStemmer().stem
    return _stem(w)


def locomo_normalize(s):
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", "".join(ch for ch in s.lower() if ch not in set(string.punctuation)))
    return " ".join(s.split())


def locomo_f1(pred, gold):
    pt = [stem(w) for w in locomo_normalize(pred).split()]
    gt = [stem(w) for w in locomo_normalize(gold).split()]
    ns = sum((Counter(pt) & Counter(gt)).values())
    if ns == 0:
        return 0.0
    p, r = ns / len(pt), ns / len(gt)
    return 2 * p * r / (p + r)


def locomo_f1_multihop(pred, gold):
    preds = [p.strip() for p in pred.split(",")]
    golds = [g.strip() for g in gold.split(",")]
    return sum(max(locomo_f1(p, g) for p in preds) for g in golds) / len(golds)


cache = {}


def get(S, point):
    """-> (groups, valid flags): groups are the bootstrap units, each a list of per-question scores, in a fixed order."""
    if (S, point) not in cache:
        tag, c = point.rsplit(".", 1)
        files = glob.glob(os.path.join(a.run_dir, "preds", tag, f"{S}.shard*.jsonl"))
        rows = sorted((r for f in files for r in read_jsonl(f)), key=lambda r: r["id"])
        assert rows, f"no predictions for {point} on {S}"
        valid = [o[f"valid_{c}"] for r in rows for o in r["official"]]
        if dataset_of(S) == "locomo":
            Q = sorted((o for r in rows for o in r["official"]), key=lambda o: o["id"])
            groups = [[(locomo_f1_multihop if o.get("extra", {}).get("category") == 1 else locomo_f1)(o[f"pred_{c}"], o["answers"][0])]
                      for o in Q]
        else:
            groups = [[o[f"f1_{c}"] for o in r["official"]] for r in rows]
        cache[(S, point)] = (groups, valid)
    return cache[(S, point)]


def mean(G, idx=None):
    idx = range(len(G)) if idx is None else idx
    s = n = 0
    for i in idx:
        s += sum(G[i])
        n += len(G[i])
    return 100 * s / n


first = (a.points or [a.pairs[0].split("-")[0]])[0]
rng = random.Random(0)
draws = [{S: [rng.randrange(len(get(S, first)[0])) for _ in range(len(get(S, first)[0]))] for S in a.sets} for _ in range(a.B)]
ci = lambda xs: (sorted(xs)[int(.025 * len(xs))], sorted(xs)[int(.975 * len(xs)) - 1])

if a.points:
    print(f"| point | {a.label} ({len(a.sets)} sets) [95% CI] | valid % | " + " | ".join(a.sets) + " |\n|" + "---|" * (len(a.sets) + 3))
    for pt in a.points:
        per = {S: mean(get(S, pt)[0]) for S in a.sets}
        lo, hi = ci([sum(mean(get(S, pt)[0], d[S]) for S in a.sets) / len(a.sets) for d in draws])
        valid = sum(100 * sum(get(S, pt)[1]) / len(get(S, pt)[1]) for S in a.sets) / len(a.sets)
        print(f"| {pt} | {sum(per.values()) / len(per):.1f} [{lo:.1f}, {hi:.1f}] | {valid:.1f} | "
              + " | ".join(f"{per[S]:.1f}" for S in a.sets) + " |")
if a.pairs:
    print(("\n" if a.points else "") + f"| pair | {a.label} difference [95% CI] | " + " | ".join(a.sets) + " |\n|" + "---|" * (len(a.sets) + 2))
    for pair in a.pairs:
        x, y = pair.split("-")
        per = {S: mean(get(S, x)[0]) - mean(get(S, y)[0]) for S in a.sets}
        lo, hi = ci([sum(mean(get(S, x)[0], d[S]) - mean(get(S, y)[0], d[S]) for S in a.sets) / len(a.sets) for d in draws])
        print(f"| {x} - {y} | {sum(per.values()) / len(per):+.1f} [{lo:+.1f}, {hi:+.1f}] | "
              + " | ".join(f"{per[S]:+.1f}" for S in a.sets) + " |")
