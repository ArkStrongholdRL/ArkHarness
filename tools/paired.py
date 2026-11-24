"""Paired comparison of two harness.sim runs on the same seeds (B3 / B6 / B7 acceptance).

  .venv\\Scripts\\python tools/paired.py logs/sim_env.json logs/sim_full.json
Prints: per-seed identity of (win, survival round, LP), the win rates and the paired win difference with its 95% interval.
"""
import json
import math
import sys


def main(a, b):
    A = {r["seed"]: r for r in json.load(open(a, encoding="utf-8"))["results"]}
    B = {r["seed"]: r for r in json.load(open(b, encoding="utf-8"))["results"]}
    seeds = sorted(set(A) & set(B))
    same = [s for s in seeds if (A[s]["win"], A[s]["survivalRound"], A[s]["lp"]) == (B[s]["win"], B[s]["survivalRound"], B[s]["lp"])]
    n = len(seeds)
    wa = sum(A[s]["win"] for s in seeds) / n
    wb = sum(B[s]["win"] for s in seeds) / n
    d = [B[s]["win"] - A[s]["win"] for s in seeds]
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / max(1, n - 1))
    half = 1.96 * sd / math.sqrt(n)
    sv = [B[s]["survivalRound"] - A[s]["survivalRound"] for s in seeds]
    print(f"seeds {n} (only in A {len(set(A) - set(B))}, only in B {len(set(B) - set(A))})")
    print(f"identical (win, round, lp): {len(same)}/{n}")
    print(f"win A {wa:.3f}  win B {wb:.3f}  paired diff B-A {mean:+.3f}  95% CI [{mean - half:+.3f}, {mean + half:+.3f}]")
    print(f"survival round diff B-A mean {sum(sv) / n:+.3f}; wins only in A {sum(1 for x in d if x < 0)}, only in B {sum(1 for x in d if x > 0)}")
    diff = [s for s in seeds if s not in same][:10]
    for s in diff:
        print(f"  seed {s}: A {A[s]} | B {B[s]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
