from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List
import json
import math
import random
from fractions import Fraction


# -----------------------------
# Core structures
# -----------------------------

@dataclass(frozen=True)
class ProblemFamily:
    family_id: str
    title: str
    template: str
    sampler: Callable[[random.Random], Dict[str, Any]]  # returns params dict


def render(template: str, params: Dict[str, Any]) -> str:
    text = template.format(**params).replace("\n", " ").strip()
    return " ".join(text.split())


# -----------------------------
# Utilities
# -----------------------------

def rint(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)

def rchoice(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]

def rfloat(rng: random.Random, lo: float, hi: float, nd: int = 3) -> float:
    return round(rng.uniform(lo, hi), nd)

def tries(rejection_sampler, max_tries: int = 50_000) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_tries):
        try:
            return rejection_sampler()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to sample a valid instance after {max_tries} tries. Last error: {last_err}")

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def cents_to_str(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents//100}.{cents%100:02d}"

def round_to_nearest_unit_cents(amount_cents: int, unit_cents: int) -> int:
    # nearest, ties .5 up
    if unit_cents <= 0:
        raise ValueError("round unit must be positive")
    return int((amount_cents + unit_cents // 2) // unit_cents * unit_cents)

def cost_up_to_units(usage: int, a: int, m: int, b: int, n: int, c: int) -> int:
    # returns pre-discount cost in cents for 'usage' units
    u1 = min(usage, m)
    u2 = min(max(usage - m, 0), n)
    u3 = max(usage - m - n, 0)
    return u1 * a + u2 * b + u3 * c

def tariff_bill_cents(
    usage: int,
    a: int, m: int,
    b: int, n: int,
    c: int,
    U_disc: int,
    d_pct: int,
    P_thresh: int,
    R_rebate: int,
    round_unit: int,
) -> int:
    pre = cost_up_to_units(usage, a, m, b, n, c)
    # discount on portion beyond U_disc (if usage > U_disc)
    if usage > U_disc:
        cost_beyond = pre - cost_up_to_units(U_disc, a, m, b, n, c)
        disc = (cost_beyond * d_pct + 50) // 100  # nearest cent
    else:
        disc = 0
    after_disc = pre - disc
    after_rebate = after_disc - (R_rebate if pre > P_thresh else 0)
    return round_to_nearest_unit_cents(after_rebate, round_unit)


# -----------------------------
# Family 13 sampler
# -----------------------------

def final_conc_pct_rounded3(
    VA: int, cA: int,
    VB: int, cB: int,
    x: int,
    s: int,
    w: int,
    u: int,
    cU: int,
) -> float:
    # All in Fractions, then return percent concentration rounded to 3 decimals
    cA_f = Fraction(cA, 100)
    cB_f = Fraction(cB, 100)
    cU_f = Fraction(cU, 100)

    vol_after_transfer = Fraction(VB + x, 1)
    solute_after_transfer = Fraction(VB, 1) * cB_f + Fraction(x, 1) * cA_f

    # spill s liters of mixed solution from B
    if s < 0 or s > VB + x:
        raise ValueError("spill infeasible")
    remaining_frac = Fraction(VB + x - s, VB + x)
    solute_after_spill = solute_after_transfer * remaining_frac
    vol_after_spill = Fraction(VB + x - s, 1)

    # add water and cU% solution
    vol_final = vol_after_spill + Fraction(w, 1) + Fraction(u, 1)
    solute_final = solute_after_spill + Fraction(u, 1) * cU_f

    conc_pct = Fraction(100, 1) * solute_final / vol_final
    return round(float(conc_pct), 3)

def sample_f13(rng: random.Random) -> Dict[str, Any]:
    # We choose a hidden true x, compute cT, and ensure uniqueness under printed rounding.
    def _one():
        VA = rint(rng, 30, 200)
        VB = rint(rng, 30, 220)

        cA = rint(rng, 30, 85)
        cB = rint(rng, 0, 60)
        if cA <= cB:
            raise ValueError("need cA>cB for monotone behavior typically")

        cU = rint(rng, 0, 70)

        x_true = rint(rng, 1, VA)
        s = rint(rng, 1, VB + x_true - 1)  # ensure positive volume after spill
        w = rint(rng, 5, 120)
        u = rint(rng, 5, 120)

        cT = final_conc_pct_rounded3(VA, cA, VB, cB, x_true, s, w, u, cU)

        # uniqueness check under 3-decimal printed cT: count x that produce same rounded cT
        matches = 0
        for x in range(1, VA + 1):
            if s <= VB + x - 1:  # spill feasibility for that x
                cT_x = final_conc_pct_rounded3(VA, cA, VB, cB, x, s, w, u, cU)
                if cT_x == cT:
                    matches += 1
                    if matches > 1:
                        raise ValueError("not unique under rounding")
        if matches != 1:
            raise ValueError("no match found")

        # Also keep spill feasible for the actual transfer (already ensured)
        return {
            "VA": VA, "cA": cA,
            "VB": VB, "cB": cB,
            "s": s, "w": w, "u": u, "cU": cU,
            "cT": cT
        }
    return tries(_one)


# -----------------------------
# Family 14 sampler
# -----------------------------

def sample_f14(rng: random.Random) -> Dict[str, Any]:
    def _one():
        m = rint(rng, 10, 80)
        n = rint(rng, 10, 80)

        # cents per unit
        a = rint(rng, 50, 500)
        b = rint(rng, 60, 700)
        c = rint(rng, 80, 900)

        # encourage piecewise impact
        if not (a < c and b != a):
            raise ValueError("resample rates")

        U_disc = rint(rng, 5, m + n + 60)
        d_pct = rint(rng, 5, 35)

        # threshold + rebate
        P_thresh = rint(rng, 10_00, 500_00)  # 10..500
        R_rebate = rint(rng, 5_00, 120_00)   # 5..120

        round_unit = rchoice(rng, [1, 5, 10, 25, 50, 100, 500])  # cents

        # choose usage in tier 2 or 3 often
        tier = rchoice(rng, [2, 3, 3])
        if tier == 2:
            usage_true = rint(rng, m + 1, m + n)
        else:
            usage_true = rint(rng, m + n + 1, m + n + 120)

        # ensure discount applies (usage > U_disc)
        if usage_true <= U_disc:
            U_disc = rint(rng, 5, usage_true - 1)

        bill_c = tariff_bill_cents(usage_true, a, m, b, n, c, U_disc, d_pct, P_thresh, R_rebate, round_unit)
        Bill = cents_to_str(bill_c)
        round_unit_str = cents_to_str(round_unit)

        # uniqueness: only one usage in a plausible range maps to Bill
        max_check = max(usage_true + 200, m + n + 200)
        cnt = 0
        for usage in range(0, max_check + 1):
            bc = tariff_bill_cents(usage, a, m, b, n, c, U_disc, d_pct, P_thresh, R_rebate, round_unit)
            if cents_to_str(bc) == Bill:
                cnt += 1
                if cnt > 1:
                    raise ValueError("non-unique usage after rounding")
        if cnt != 1:
            raise ValueError("no usage produces the bill")

        return {
            "a": cents_to_str(a), "m": m,
            "b": cents_to_str(b), "n": n,
            "c": cents_to_str(c),
            "U": U_disc, "d": d_pct,
            "P": cents_to_str(P_thresh),
            "R": cents_to_str(R_rebate),
            "round_unit": round_unit_str,
            "Bill": Bill
        }
    return tries(_one)


# -----------------------------
# Family 15 sampler
# -----------------------------

def sample_f15(rng: random.Random) -> Dict[str, Any]:
    def _one():
        x = rint(rng, 8, 70)
        y = rint(rng, 1, x - 2)
        S = x + y
        D = x - y
        P = x * y
        # ensure override is informative: choose T between y and x-1 so only x exceeds T
        T = rint(rng, y, x - 1)
        if not (x > T and y <= T):
            raise ValueError("override not uniquely identifying x")
        return {"S": S, "D": D, "T": T, "P": P}
    return tries(_one)


# -----------------------------
# Family 16 sampler
# -----------------------------

def sample_f16(rng: random.Random) -> Dict[str, Any]:
    def _one():
        a0 = rint(rng, 0, 40)

        r1 = rchoice(rng, [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35])
        r2 = rchoice(rng, [1.10, 1.15, 1.20, 1.25, 1.35, 1.45, 1.55, 1.65])

        c1 = rint(rng, 0, 15)
        c2 = rint(rng, 0, 20)

        T = rint(rng, 15, 80)
        H = rint(rng, T + 20, T + 220)

        if a0 > H:
            raise ValueError("already above H")

        # simulate, enforce monotone increasing and crossing
        a = float(a0)
        crossed_T = (a > T)
        crossed_T_at = None
        max_steps = 600
        for n in range(max_steps):
            prev = a
            if a <= T:
                a = r1 * a + c1
            else:
                a = r2 * a + c2
                if crossed_T_at is None:
                    crossed_T_at = n  # first step index where rule2 used
            if a < prev - 1e-9:
                raise ValueError("not monotone increasing")
            if a > T and not crossed_T:
                crossed_T = True
            if a > H:
                N = n + 1
                if N < 2:
                    raise ValueError("too trivial")
                if not crossed_T:
                    raise ValueError("never crossed T (no regime switch)")
                if N > 250:
                    raise ValueError("too long")
                return {
                    "a0": a0, "r1": round(r1, 2), "c1": c1,
                    "T": T, "r2": round(r2, 2), "c2": c2,
                    "H": H
                }
        raise ValueError("did not exceed H in time")
    return tries(_one)


# -----------------------------
# Family 17 sampler
# -----------------------------

def sample_f17(rng: random.Random) -> Dict[str, Any]:
    def _one():
        TA = rint(rng, 4, 20)
        TB = rint(rng, 4, 20)

        rateA = 1.0 / TA
        rateB = 1.0 / TB
        rateAB = rateA + rateB

        t1 = rint(rng, 1, 12)
        frac_done = t1 * rateAB
        if not (0.10 < frac_done < 0.92):
            raise ValueError("first phase too small/large")

        # choose p so we are not near the boundary
        # branch determined by whether frac_done < p/100
        # pick one of the branches randomly but still embed both parameters
        branch = rchoice(rng, ["addC", "fatigue"])

        if branch == "addC":
            p = rint(rng, int(math.ceil(frac_done * 100)) + 5, min(95, int(math.ceil(frac_done * 100)) + 40))
        else:
            p = rint(rng, max(5, int(math.floor(frac_done * 100)) - 40), int(math.floor(frac_done * 100)) - 5)

        if not (1 <= p <= 99):
            raise ValueError("bad p")

        # parameters for both clauses
        eC = rchoice(rng, [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
        d = rint(rng, 5, 40)
        B = rint(rng, 50, 500)

        # compute completion time deterministically with the actual triggered branch
        if frac_done < p / 100.0:
            # add C
            rateC = eC * rateA
            total_rate = rateA + rateB + rateC
        else:
            # fatigue
            rateA2 = (1.0 - d / 100.0) * rateA
            total_rate = rateA2 + rateB

        remaining = (1.0 - frac_done) / total_rate
        if remaining <= 0:
            raise ValueError("already finished or invalid")
        total_time = t1 + remaining
        if total_time > 60:
            raise ValueError("too long")

        # force bonus to trigger (avoid conditional ambiguity)
        Tbonus = math.ceil(total_time) + rint(rng, 1, 5)

        return {
            "TA": TA, "TB": TB, "t1": t1,
            "p": p, "eC": eC, "d": d,
            "Tbonus": Tbonus, "B": B
        }
    return tries(_one)


# -----------------------------
# Family 18 sampler
# -----------------------------

def simulate_phase1_inventory(N0: int, D1: int, s1: int, sh_pct: int) -> tuple[int, int]:
    inv = N0
    sold_total = 0
    for _ in range(D1):
        sell = min(s1, inv)
        inv -= sell
        sold_total += sell
        inv = int(math.floor(inv * (1.0 - sh_pct / 100.0)))
        if inv < 0:
            inv = 0
    return inv, sold_total

def sample_f18(rng: random.Random) -> Dict[str, Any]:
    def _one():
        N0 = rint(rng, 120, 900)
        c0 = rint(rng, 200, 2500)  # cents
        D1 = rint(rng, 5, 18)
        s1 = rint(rng, 1, max(2, min(35, N0 // D1)))
        sh = rint(rng, 1, 10)
        p1 = rint(rng, c0 + 100, c0 + 4000)

        inv_after, sold1 = simulate_phase1_inventory(N0, D1, s1, sh)
        if sold1 <= 0:
            raise ValueError("no sales phase1")

        N1 = rint(rng, 60, 700)
        c1 = rint(rng, 200, 3000)
        D2 = rint(rng, 5, 18)

        inv2 = inv_after + N1
        if inv2 < 20:
            raise ValueError("too little inventory for phase2")
        s2 = rint(rng, 1, max(2, min(35, inv2 // D2)))
        sold2 = s2 * D2
        if sold2 <= 0 or sold2 > inv2:
            raise ValueError("phase2 infeasible sales")

        # choose p2 (unknown) implicitly by choosing a feasible margin M and computing M from a random p2
        # To keep things consistent, pick p2 first (in cents), compute M with 4 decimals.
        p2_hidden = rint(rng, max(50, c1 + 50), c1 + 5000)

        cost_total = N0 * c0 + N1 * c1
        rev1 = sold1 * p1
        rev2 = sold2 * p2_hidden
        profit = (rev1 + rev2) - cost_total
        if profit <= 0:
            raise ValueError("nonpositive profit")
        M = round(100.0 * profit / cost_total, 4)

        # sanity range
        if not (1.0 <= M <= 80.0):
            raise ValueError("margin too extreme")

        return {
            "N0": N0, "c0": cents_to_str(c0),
            "D1": D1, "s1": s1, "p1": cents_to_str(p1),
            "sh": sh,
            "N1": N1, "c1": cents_to_str(c1),
            "D2": D2, "s2": s2,
            "M": M
        }
    return tries(_one)


# -----------------------------
# Family 19 sampler
# -----------------------------

def sample_f19(rng: random.Random) -> Dict[str, Any]:
    def _one():
        # choose weights from a small discrete set for clean decimals
        wE = rchoice(rng, [0.30, 0.35, 0.40, 0.45, 0.50])
        wQ = rchoice(rng, [0.10, 0.15, 0.20, 0.25, 0.30])
        wP = round(1.0 - wE - wQ, 2)
        if wP <= 0.05:
            raise ValueError("bad weights")

        Delta = rchoice(rng, [0.05, 0.10, 0.15])
        if wP - Delta <= 0.01 or wE + Delta >= 0.99:
            raise ValueError("bad Delta")

        T = rint(rng, 40, 85)

        E_true = rint(rng, 0, T - 1)
        pval = rint(rng, 40, 100)
        qval = rint(rng, 40, 100)

        wE2 = wE + Delta
        wP2 = wP - Delta
        wQ2 = wQ

        F = wE2 * E_true + wP2 * pval + wQ2 * qval
        F = round(F, 2)

        # ensure not near boundary
        if abs(E_true - T) < 1:
            raise ValueError("too close to threshold")

        return {
            "wE": wE, "wP": wP, "wQ": wQ,
            "T": T, "Delta": Delta,
            "pval": pval, "qval": qval,
            "F": F
        }
    return tries(_one)


# -----------------------------
# Family 20 sampler
# -----------------------------

def sample_f20(rng: random.Random) -> Dict[str, Any]:
    def _one():
        X = rfloat(rng, 500.0, 8000.0, 2)

        rAB = rfloat(rng, 0.4, 2.2, 4)
        rCA = rfloat(rng, 0.4, 2.2, 4)

        f1 = rfloat(rng, 0.5, 6.0, 2)
        f3 = rfloat(rng, 0.5, 6.0, 2)

        B_amt = X * rAB * (1.0 - f1 / 100.0)
        if B_amt <= 5.0:
            raise ValueError("too small B amount")

        F2 = rfloat(rng, 1.0, min(0.15 * B_amt, 200.0), 2)  # in currency B
        if B_amt - F2 <= 1.0:
            raise ValueError("flat fee too large")

        denom = (B_amt - F2) * rCA * (1.0 - f3 / 100.0)  # Y = denom * rBC
        if denom <= 0:
            raise ValueError("bad denom")

        # choose a slightly unprofitable baseline and epsilon that flips profitability
        ratio = rng.uniform(0.97, 0.995)
        Y_target = X * ratio
        rBC_hidden = Y_target / denom

        # epsilon to flip: need denom*(rBC+eps) > X => eps > (X - Y)/denom
        eps_min = (X - Y_target) / denom
        eps = eps_min + rng.uniform(0.01 * rBC_hidden, 0.05 * rBC_hidden)

        # keep epsilon reasonable
        if eps <= 0 or eps > 0.25 * rBC_hidden or eps > 2.0:
            raise ValueError("epsilon unreasonable; resample")

        # print Y with enough precision
        Y = round(Y_target, 4)
        eps_out = round(eps, 6)

        return {
            "X": round(X, 2),
            "rAB": rAB, "f1": f1,
            "F2": round(F2, 2),
            "rCA": rCA, "f3": f3,
            "Y": Y,
            "eps": eps_out
        }
    return tries(_one)


# -----------------------------
# Family 21 sampler
# -----------------------------

def sample_f21(rng: random.Random) -> Dict[str, Any]:
    def _one():
        Q = rint(rng, 60, 420)

        min2 = rint(rng, max(1, int(0.10 * Q)), int(0.45 * Q))
        cap1 = rint(rng, int(0.45 * Q), int(0.95 * Q))

        if Q - min2 > cap1:
            raise ValueError("infeasible due to cap1/min2")

        k1 = rint(rng, 10, max(11, min(cap1, Q) - 5))
        k2 = rint(rng, 10, max(11, min(Q, Q - 5) - 5))

        a1 = rint(rng, 2, 8)
        b1 = rint(rng, 8, 18)
        a2 = rint(rng, 3, 12)
        b2 = rint(rng, 1, 9)

        # encourage cross-over structure
        if not (a1 < b1 and a2 > b2 and b2 < b1):
            raise ValueError("not interesting enough")

        return {
            "Q": Q,
            "a1": a1, "k1": k1, "b1": b1, "cap1": cap1,
            "a2": a2, "k2": k2, "b2": b2, "min2": min2
        }
    return tries(_one)


# -----------------------------
# Family 22 sampler
# -----------------------------

def sample_f22(rng: random.Random) -> Dict[str, Any]:
    def _one():
        a = rchoice(rng, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        tv = rint(rng, -6, 6)
        b = -2 * a * tv
        c = rint(rng, -30, 30)

        # choose distinct t points not all near each other
        ts = set()
        while len(ts) < 3:
            ts.add(rint(rng, -8, 8))
        t1, t2, t3 = sorted(ts)
        if len({t1, t2, t3}) != 3:
            raise ValueError("need distinct t")

        def f(t: int) -> int:
            return a * t * t + b * t + c

        y1, y2, y3 = f(t1), f(t2), f(t3)

        return {"t1": t1, "y1": y1, "t2": t2, "y2": y2, "t3": t3, "y3": y3, "tv": tv}
    return tries(_one)


# -----------------------------
# Families (13–22)
# -----------------------------

FAMILIES: List[ProblemFamily] = [
    ProblemFamily(
        "f13",
        "Two-mixture with loss, refill, and purity constraint",
        "Tank A contains {VA} liters of solution at {cA}% solute; Tank B contains {VB} liters at {cB}%. You transfer x liters from A to B and mix thoroughly, then spill {s} liters from B. Next you add {w} liters of pure water to B and {u} liters of {cU}% solution to B. After all operations, B’s concentration is exactly {cT}%. Find x.",
        sample_f13,
    ),
    ProblemFamily(
        "f14",
        "Piecewise tariff + tiered discount with rounding",
        "A service charges {a} per unit for the first {m} units, {b} per unit for the next {n} units, and {c} per unit beyond that. If total usage exceeds {U}, a discount of {d}% applies only to the portion beyond {U}; additionally, if the pre-discount bill exceeds {P}, a flat rebate {R} is subtracted. The bill is rounded to the nearest {round_unit}. Given final bill {Bill}, find the total usage.",
        sample_f14,
    ),
    ProblemFamily(
        "f15",
        "System of equations with conditional override rule",
        "Two numbers x and y satisfy: (i) x + y = {S}, (ii) x − y = {D} unless x exceeds {T}, in which case condition (ii) is replaced by x·y = {P}. You are told the override did occur. Find x and y.",
        sample_f15,
    ),
    ProblemFamily(
        "f16",
        "Recurrence with regime switch based on threshold",
        "A sequence starts at a0 = {a0}. For n ≥ 0, define a_{{n+1}} = {r1}·a_n + {c1} if a_n ≤ {T}, and a_{{n+1}} = {r2}·a_n + {c2} otherwise. Let N be the smallest index where a_N first exceeds {H}. Compute N and a_N.",
        sample_f16,
    ),
    ProblemFamily(
        "f17",
        "Work-rate with efficiency drop + bonus completion clause",
        "Worker A alone completes a job in {TA} hours and Worker B alone in {TB} hours. They work together for {t1} hours. If by then less than {p}% of the job is done, they add Worker C who works at {eC} times A’s rate; otherwise A’s efficiency drops by {d}% for the remaining time due to fatigue. Additionally, if the job finishes before {Tbonus} hours total, they receive a bonus {B} split proportional to individual hours worked. Find the total completion time and A’s bonus share.",
        sample_f17,
    ),
    ProblemFamily(
        "f18",
        "Inventory with shrinkage, restock, and target margin",
        "A store starts with {N0} units costing {c0} each. Each day for {D1} days it sells {s1} units at price {p1}; unsold inventory shrinks by {sh}% daily. Then it restocks {N1} units at cost {c1} and changes selling price to p2 for {D2} days, selling {s2} units per day (cannot exceed inventory). Given the total profit margin over the whole period is {M}% relative to total cost, find p2.",
        sample_f18,
    ),
    ProblemFamily(
        "f19",
        "Weighted average with constraint-dependent weights",
        "A final score is computed from Exam E, Project P, and Quiz average Q with weights wE={wE}, wP={wP}, wQ={wQ} (summing to 1). However, if E is below {T}, the exam weight increases by {Delta} and the project weight decreases by {Delta} (quiz unchanged). A student has P={pval}, Q={qval}, and final score {F}. Given the adjustment did occur, find E.",
        sample_f19,
    ),
    ProblemFamily(
        "f20",
        "Multi-currency exchange with fees and arbitrage check",
        "You start with {X} in currency A. You exchange to currency B at rate {rAB} with fee {f1}%, then to currency C at rate rBC with a flat fee {F2} (in currency B), then back to A at rate {rCA} with fee {f3}%. If the final amount in A is {Y}, determine the missing rate rBC and state whether the cycle is profitable if rBC increases by {eps}.",
        sample_f20,
    ),
    ProblemFamily(
        "f21",
        "Piecewise linear cost minimization with constraint coupling",
        "You must produce {Q} units using Machine 1 and Machine 2. Machine 1 cost is {a1} per unit up to {k1} units then {b1} per unit beyond; Machine 2 cost is {a2} per unit up to {k2} units then {b2} per unit beyond. Additionally, Machine 1 output cannot exceed {cap1}, and at least {min2} units must come from Machine 2. Find the minimum total cost and the optimal split.",
        sample_f21,
    ),
    ProblemFamily(
        "f22",
        "Polynomial parameter fit with extra constraint and integer solution",
        "A quadratic f(t)=a t^2 + b t + c satisfies f({t1})={y1}, f({t2})={y2}, and f({t3})={y3}. In addition, the vertex occurs at t={tv} (i.e., −b/(2a)={tv}) and you are told a is an integer. Find (a,b,c).",
        sample_f22,
    ),
]


# -----------------------------
# Generation + saving (same layout as first probstatgen)
# -----------------------------

def generate_instances(rng: random.Random, family: ProblemFamily, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    while len(out) < n:
        params = family.sampler(rng)
        key = tuple(sorted(params.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        text = render(family.template, params)
        out.append({"family_id": family.family_id, "title": family.title, "params": params, "text": text})
    return out


def main(out_dir: str = "out_alg", n_per_family: int = 10, seed: int = 12345) -> None:
    rng = random.Random(seed)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    manifest = [{"family_id": f.family_id, "title": f.title, "template": f.template} for f in FAMILIES]
    (outp / "families_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    all_rows: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        rows = generate_instances(rng, fam, n_per_family)
        for i, r in enumerate(rows, start=1):
            r["problem_id"] = f"{fam.family_id}_{i:02d}"
        all_rows.extend(rows)
        write_jsonl(outp / "by_family" / f"{fam.family_id}.jsonl", rows)

    write_jsonl(outp / "problems.jsonl", all_rows)

    txt_lines = [f"[{r['problem_id']}] {r['text']}" for r in all_rows]
    (outp / "problems.txt").write_text("\n\n".join(txt_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
