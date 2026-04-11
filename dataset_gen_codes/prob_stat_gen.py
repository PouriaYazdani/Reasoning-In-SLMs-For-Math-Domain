from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple
import json
import math
import random


# -----------------------------
# Core data structures
# -----------------------------

@dataclass(frozen=True)
class ProblemFamily:
    family_id: str
    title: str
    template: str
    sampler: Callable[[random.Random], Dict[str, Any]]  # returns params dict


def render(template: str, params: Dict[str, Any]) -> str:
    # Keep single paragraph: collapse any newlines.
    text = template.format(**params).replace("\n", " ").strip()
    # Collapse repeated whitespace
    text = " ".join(text.split())
    return text


# -----------------------------
# Small utilities
# -----------------------------

def rint(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)

def rfloat(rng: random.Random, lo: float, hi: float, nd: int = 3) -> float:
    return round(rng.uniform(lo, hi), nd)

def rprob(rng: random.Random, lo: float = 0.05, hi: float = 0.95, nd: int = 3) -> float:
    return rfloat(rng, lo, hi, nd)

def choose_distinct_ints(rng: random.Random, lo: int, hi: int, k: int) -> List[int]:
    if hi - lo + 1 < k:
        raise ValueError("Range too small for distinct sampling.")
    vals = list(range(lo, hi + 1))
    rng.shuffle(vals)
    return sorted(vals[:k])

def poisson_cdf(k: int, lam: float) -> float:
    # P(X <= k) for Poisson(lam). Stable enough for small k, moderate lam.
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    s = term
    for i in range(1, k + 1):
        term *= lam / i
        s += term
    return min(1.0, max(0.0, s))

def tries(rejection_sampler: Callable[[], Dict[str, Any]], max_tries: int = 10_000) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_tries):
        try:
            return rejection_sampler()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to sample a valid instance after {max_tries} tries. Last error: {last_err}")


# -----------------------------
# Family samplers (respect constraints)
# -----------------------------

H_SET = ["N", "N^2", "1/N", "log(N)"]


def sample_f01(rng: random.Random) -> Dict[str, Any]:
    # Stop-on-K reds with truncation; condition ended by K reds.
    def _one():
        R = rint(rng, 4, 25)
        B = rint(rng, 4, 25)
        K = rint(rng, 1, min(R, 8))
        # Ensure truncation is meaningful and feasible
        S = rint(rng, K + 1, min(R + B - 1, 40))
        # N must allow K-th red at draw N: need N-K <= B and N<=S
        Nmax = min(S, R + B, B + K)
        N = rint(rng, K, Nmax)
        # P(stop by K reds) > 0 requires S >= K (true) and R >= K (true)
        return {"R": R, "B": B, "K": K, "S": S, "N": N}
    return tries(_one)


def sample_f02(rng: random.Random) -> Dict[str, Any]:
    # Urn draw then branch-dependent update then second draw conditional on X=x.
    def _one():
        R0 = rint(rng, 8, 30)
        B0 = rint(rng, 8, 30)
        d1 = rint(rng, 3, min(12, R0 + B0 - 1))

        # Feasible x range in first draw
        lo = max(0, d1 - B0)
        hi = min(d1, R0)
        if lo >= hi:
            raise ValueError("No variability for X in first draw.")

        # Threshold T must be interior so both branches are possible in principle
        T = rint(rng, lo + 1, hi)

        # Condition on a specific x
        x = rint(rng, lo, hi)

        # Define both branch params (even if one won't be used under the conditioning)
        aR = rint(rng, 1, 6)
        aB = rint(rng, 1, 6)

        remR = R0 - x
        remB = B0 - (d1 - x)

        # If x < T, removal branch triggers in the conditioned event: must be feasible
        if x < T:
            rR = rint(rng, 0, min(6, remR))
            rB = rint(rng, 0, min(6, remB))
            # avoid totally trivial "remove 0,0" too often
            if rR == 0 and rB == 0 and (remR > 0 or remB > 0):
                rR = min(1, remR)
        else:
            # removal params still shown in statement; pick small feasible defaults
            rR = rint(rng, 0, min(4, remR))
            rB = rint(rng, 0, min(4, remB))

        # Updated urn given x
        if x >= T:
            R1 = remR + aR
            B1 = remB + aB
        else:
            R1 = remR - rR
            B1 = remB - rB
            if R1 < 0 or B1 < 0:
                raise ValueError("Removal infeasible.")

        total1 = R1 + B1
        if total1 <= 0:
            raise ValueError("Empty urn after update.")

        d2 = rint(rng, 1, min(12, total1))
        y_lo = max(0, d2 - B1)
        y_hi = min(d2, R1)
        if y_lo > y_hi:
            raise ValueError("No feasible y in second draw.")
        y = rint(rng, y_lo, y_hi)

        return {
            "R0": R0, "B0": B0, "d1": d1, "x": x, "T": T,
            "aR": aR, "aB": aB, "rR": rR, "rB": rB,
            "d2": d2, "y": y
        }
    return tries(_one)


def sample_f03(rng: random.Random) -> Dict[str, Any]:
    # Partial revelation: X in M_SET, then probability second batch has >= q red.
    def _one():
        R = rint(rng, 8, 30)
        B = rint(rng, 8, 30)
        n = rint(rng, 3, min(14, R + B - 3))
        # feasible X range for reds in first n
        lo = max(0, n - B)
        hi = min(n, R)
        if hi - lo + 1 < 3:
            raise ValueError("Not enough feasible red-count values for M_SET size 3.")
        t = rint(rng, 1, min(12, R + B - n))
        # To ensure conditional tail event has positive probability for at least one m,
        # require m <= R - q for some m in M_SET.
        max_m_for_q = min(hi, R - 1)  # start with q>=1
        if max_m_for_q < lo:
            raise ValueError("Cannot ensure q>=1 with any feasible m.")
        q = rint(rng, 1, min(t, R - lo))  # ensure at least for m=lo, remaining reds >= q
        max_m_for_q = min(hi, R - q)
        if max_m_for_q - lo + 1 < 3:
            raise ValueError("Not enough m values that keep remaining reds >= q.")
        ms = choose_distinct_ints(rng, lo, max_m_for_q, 3)
        M_SET = "{" + ",".join(map(str, ms)) + "}"
        return {"R": R, "B": B, "n": n, "M_SET": M_SET, "t": t, "q": q}
    return tries(_one)


def sample_f04(rng: random.Random) -> Dict[str, Any]:
    def _one():
        pi = rprob(rng)
        pA = rprob(rng)
        pB = rprob(rng)
        sens = rprob(rng)
        fp = rprob(rng)
        r = rint(rng, 1, 6)
        u = rint(rng, 1, r + 1)
        cT = rint(rng, 1, 25)
        cD = rint(rng, 10, 120)
        cF = rint(rng, 20, 250)
        return {"pi": pi, "pA": pA, "pB": pB, "sens": sens, "fp": fp, "r": r, "u": u, "cT": cT, "cD": cD, "cF": cF}
    return tries(_one)


def sample_f05(rng: random.Random) -> Dict[str, Any]:
    def _one():
        L = rint(rng, 4, 12)
        s = rint(rng, 1, L - 1)
        # choose p,q on a grid to reduce accidental p+q>=1
        grid = [i / 100 for i in range(5, 91, 5)]
        p = rng.choice(grid)
        q = rng.choice(grid)
        if not (p > 0 and q > 0 and p + q < 1):
            raise ValueError("Invalid p,q.")
        alpha = rint(rng, -3, 5)
        beta = rint(rng, 0, 12)
        g_expr = f"g(i) = {alpha}*i + {beta}"
        F = rint(rng, 0, 80)
        return {"L": L, "s": s, "p": round(p, 2), "q": round(q, 2), "g_expr": g_expr, "F": F}
    return tries(_one)


def sample_f06(rng: random.Random) -> Dict[str, Any]:
    def _one():
        p = rprob(rng)
        S = rint(rng, 6, 25)
        k = rint(rng, 1, S - 1)
        h = rng.choice(H_SET)
        return {"p": p, "k": k, "S": S, "h": h}
    return tries(_one)


def sample_f07(rng: random.Random) -> Dict[str, Any]:
    def _one():
        C = rint(rng, 4, 18)
        m = rint(rng, 1, C)
        b = rint(rng, 1, 5)
        return {"C": C, "m": m, "b": b}
    return tries(_one)


def sample_f08(rng: random.Random) -> Dict[str, Any]:
    def _one():
        A = rint(rng, 12, 45)
        O = rint(rng, 12, 45)
        t = rint(rng, 2, 9)
        r = rint(rng, 1, t)
        blocks = rint(rng, 3, 8)
        S = t * blocks  # enforce t|S
        if A + O < S:
            raise ValueError("Need enough items to complete S draws even if no returns happen.")
        # Non-extreme u; still within feasible apples drawn so far bounds.
        u = rint(rng, 1, min(A, S - 1))
        return {"A": A, "O": O, "t": t, "u": u, "r": r, "S": S}
    return tries(_one)


def sample_f09(rng: random.Random) -> Dict[str, Any]:
    def _one():
        m = rint(rng, 2, 6)
        t = rint(rng, 1, 3)
        d = rint(rng, 3, 12)

        min_total1 = m + m * t + d
        min_total2 = m + m * t

        total1 = rint(rng, min_total1, min_total1 + 30)
        total2 = rint(rng, min_total2, min_total2 + 30)

        R1 = rint(rng, 1, total1 - 1)
        B1 = total1 - R1
        R2 = rint(rng, 1, total2 - 1)
        B2 = total2 - R2

        x = rint(rng, 0, d)
        return {"R1": R1, "B1": B1, "R2": R2, "B2": B2, "m": m, "t": t, "d": d, "x": x}
    return tries(_one)


def sample_f10(rng: random.Random) -> Dict[str, Any]:
    def _one():
        NA = rint(rng, 80, 800)
        NB = rint(rng, 80, 800)
        nA = rint(rng, 8, min(80, NA - 1))
        nB = rint(rng, 8, min(80, NB - 1))

        muA = rfloat(rng, -10, 10, 2)
        muB = rfloat(rng, -10, 10, 2)
        sA2 = rfloat(rng, 0.5, 25, 2)
        sB2 = rfloat(rng, 0.5, 25, 2)

        delta = rfloat(rng, -5, 5, 2)
        if abs(delta) < 0.1:
            delta = 1.0

        # Pick τ near the expected combined sample mean to avoid degeneracy.
        mu_samp = (nA * muA + nB * muB) / (nA + nB)
        se = math.sqrt(sA2 / nA + sB2 / nB)
        z = rng.uniform(-0.5, 0.5)
        tau = round(mu_samp + z * se, 3)

        return {
            "NA": NA, "NB": NB, "nA": nA, "nB": nB,
            "muA": muA, "muB": muB, "sA2": sA2, "sB2": sB2,
            "delta": delta, "tau": tau
        }
    return tries(_one)


def sample_f11(rng: random.Random) -> Dict[str, Any]:
    def _one():
        T = rint(rng, 1, 6)
        U = rint(rng, 1, 6)

        lam1 = rfloat(rng, 0.5, 12.0, 2)
        lam2 = rfloat(rng, 0.5, 14.0, 2)
        lam3 = rfloat(rng, 0.5, 14.0, 2)

        mean1 = lam1 * T
        # choose k so P(N1>=k) is not near 0/1 using Poisson approx check
        k_lo = max(1, int(math.floor(0.6 * mean1)))
        k_hi = max(k_lo, int(math.ceil(1.4 * mean1 + 2)))
        k = rint(rng, k_lo, k_hi)

        p_branch = 1.0 - poisson_cdf(k - 1, mean1)
        if not (0.15 <= p_branch <= 0.85):
            raise ValueError("Branch probability too extreme; resample.")

        exp_total_mid = mean1 + U * (0.5 * (lam2 + lam3))
        M_lo = max(0, int(math.floor(0.75 * exp_total_mid)))
        M_hi = max(M_lo + 1, int(math.ceil(1.25 * exp_total_mid + 3)))
        M = rint(rng, M_lo, M_hi)

        return {"lam1": lam1, "lam2": lam2, "lam3": lam3, "T": T, "U": U, "k": k, "M": M}
    return tries(_one)


def sample_f12(rng: random.Random) -> Dict[str, Any]:
    def _one():
        sigma = rfloat(rng, 0.5, 10.0, 2)
        n0 = rint(rng, 6, 35)
        n1 = rint(rng, 5, 40)
        n2 = rint(rng, 5, 40)
        c = rng.uniform(0.8, 1.2)
        v0 = round((sigma ** 2) * c, 4)
        alpha = rng.choice([0.10, 0.05, 0.02, 0.01])
        return {"sigma": sigma, "n0": n0, "n1": n1, "n2": n2, "v0": v0, "alpha": alpha}
    return tries(_one)


# -----------------------------
# Family registry (templates are single-paragraph)
# -----------------------------

FAMILIES: List[ProblemFamily] = [
    ProblemFamily(
        "f01",
        "Stop-on-K successes with conditioned stop cause",
        "A box has {R} red and {B} black cards. Cards are drawn without replacement until either {K} reds appear or {S} total draws are made, whichever happens first. You are told the process ended because {K} reds appeared (not because the draw limit was hit). What is the probability that exactly {N} draws were needed?",
        sample_f01,
    ),
    ProblemFamily(
        "f02",
        "Urn with branch-dependent state update, then second-stage target",
        "An urn starts with {R0} red and {B0} blue balls. You draw {d1} balls without replacement and observe {x} reds. If {x} ≥ {T}, you add {aR} red and {aB} blue; otherwise you remove {rR} red and {rB} blue (if possible). Then you draw {d2} balls from the updated urn. Compute P(second draw has exactly {y} red | first draw had exactly {x} red).",
        sample_f02,
    ),
    ProblemFamily(
        "f03",
        "Partial revelation then conditional probability",
        "A deck has {R} red and {B} black cards. You draw {n} cards without replacement and only learn that the number of reds is in the set {M_SET}. Then you draw {t} more cards from the remaining deck. What is the probability that the second batch contains at least {q} red given this partial information?",
        sample_f03,
    ),
    ProblemFamily(
        "f04",
        "Two-source mixture with posterior after noisy test, then expected cost",
        "A component is sourced from Supplier A with probability {pi} and Supplier B otherwise. Defect rate is {pA} for A and {pB} for B. A test returns “fail” with sensitivity {sens} and false-positive rate {fp}. You test one component; if it shows “fail,” you do {r} retests and discard the component if at least {u} of all {r_plus} tests indicate “fail.” Each test costs {cT}, discarding costs {cD}, and keeping a defective component costs {cF}. Compute the expected total cost.",
        # Wrap sampler to add r_plus for rendering
        lambda rng: (lambda d: {**d, "r_plus": d["r"] + 1})(sample_f04(rng)),
    ),
    ProblemFamily(
        "f05",
        "Markov chain with absorbing barrier and reward accumulation",
        "A token moves daily among states {{0,1,2,…,L}} with L = {L}. From state i (1 ≤ i ≤ L−1), it goes to i+1 with probability {p}, to i−1 with probability {q}, and stays with probability 1−p−q. States 0 and L are absorbing. Starting from s = {s}, you earn g(i) coins each day you are in state i (including the start day) until absorption, where {g_expr}. After absorption you pay a one-time fee {F} if absorbed at L (otherwise 0). Compute the expected net coins.",
        sample_f05,
    ),
    ProblemFamily(
        "f06",
        "Conditional expectation of nonlinear function of a stopping time",
        "Independent trials succeed with probability {p}. You run trials until you obtain {k} successes, but you stop early at {S} trials if not yet reached. Let N be the number of trials actually performed. Given that you did reach {k} successes before {S}, compute E[h(N) | reached k before S] for h(N) = {h}.",
        sample_f06,
    ),
    ProblemFamily(
        "f07",
        "Sequential sampling with replacement and bonus draws",
        "There are {C} coupon types. Each purchase yields one coupon uniformly at random. Whenever you obtain a new type for the first time, you immediately receive {b} bonus coupons (each uniformly random and independent). You stop when you have collected at least {m} distinct types. Compute the expected number of purchases.",
        sample_f07,
    ),
    ProblemFamily(
        "f08",
        "Without-replacement with midstream reshuffle rule",
        "A bag contains {A} apples and {O} oranges. You draw items without replacement. After every {t} draws, if the number of apples drawn so far is at least {u}, you return {r} uniformly random drawn items to the bag and continue; otherwise you discard the last {r} drawn items permanently. You stop after {S} total draws. What is the expected number of apples drawn?",
        sample_f08,
    ),
    ProblemFamily(
        "f09",
        "Two-urn transfer with conditional switch of transfer direction",
        "Urn 1 has {R1} red and {B1} blue; Urn 2 has {R2} red and {B2} blue. You perform {m} rounds: in each round, draw one ball from each urn (without replacement within each urn), compare colors; if they match you transfer {t} balls from Urn 1 to Urn 2 uniformly at random from Urn 1’s remaining contents, otherwise transfer {t} balls from Urn 2 to Urn 1. After {m} rounds, you draw {d} balls from Urn 1. Compute the probability of getting exactly {x} red.",
        sample_f09,
    ),
    ProblemFamily(
        "f10",
        "Stratified sampling with unequal weights and post-stratum adjustment",
        "A population has two strata: Stratum A size {NA} with mean {muA} and variance {sA2}, and Stratum B size {NB} with mean {muB} and variance {sB2}. You sample {nA} from A and {nB} from B without replacement. Then you discover a measurement bias: all A observations should be increased by {delta} only if the overall sample mean is below {tau}, otherwise no correction is applied. Compute the expected corrected overall mean.",
        sample_f10,
    ),
    ProblemFamily(
        "f11",
        "Multi-phase Poisson process with threshold-triggered rate change",
        "Calls arrive as a Poisson process with rate {lam1} per hour for the first {T} hours. If at least {k} calls arrived in that period, the rate switches to {lam2} for the next {U} hours; otherwise it switches to {lam3}. Compute the probability that total calls over {T_plus_U} hours exceeds {M}.",
        lambda rng: (lambda d: {**d, "T_plus_U": d["T"] + d["U"]})(sample_f11(rng)),
    ),
    ProblemFamily(
        "f12",
        "Confidence interval width under adaptive sample size rule",
        "You measure a process with unknown mean and known standard deviation σ = {sigma}. You take {n0} initial samples. If the sample variance exceeds {v0}, you take an additional {n1} samples; otherwise take {n2}. Using a (1−α) normal confidence interval for the mean with α = {alpha}, compute the expected confidence-interval width.",
        sample_f12,
    ),
]


# -----------------------------
# Dataset generation + saving
# -----------------------------

def generate_instances(
    rng: random.Random,
    family: ProblemFamily,
    n: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()

    while len(out) < n:
        params = family.sampler(rng)
        # dedupe by a stable tuple of items
        key = tuple(sorted(params.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)

        text = render(family.template, params)
        out.append({"family_id": family.family_id, "title": family.title, "params": params, "text": text})

    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(
    out_dir: str = "out_stat",
    n_per_family: int = 10,
    seed: int = 12345,
) -> None:
    rng = random.Random(seed)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Save manifest of families (easy to version-control/edit later)
    manifest = [
        {"family_id": fam.family_id, "title": fam.title, "template": fam.template}
        for fam in FAMILIES
    ]
    (outp / "families_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    all_rows: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        rows = generate_instances(rng, fam, n_per_family)

        # Add a problem_id that is stable within the output
        for i, r in enumerate(rows, start=1):
            r["problem_id"] = f"{fam.family_id}_{i:02d}"

        all_rows.extend(rows)
        write_jsonl(outp / "by_family" / f"{fam.family_id}.jsonl", rows)

    # Write combined dataset
    write_jsonl(outp / "problems.jsonl", all_rows)

    # Convenience: also write a plain text file
    txt_lines = []
    for r in all_rows:
        txt_lines.append(f"[{r['problem_id']}] {r['text']}")
    (outp / "problems.txt").write_text("\n\n".join(txt_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
