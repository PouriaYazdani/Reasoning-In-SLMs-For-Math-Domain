# nt_fin_gen.py
# Python 3.10+
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple, Optional
import json
import math
import random


# -----------------------------
# Core structures
# -----------------------------

@dataclass(frozen=True)
class ProblemFamily:
    family_id: str
    title: str
    template: str
    sampler: Callable[[random.Random], Dict[str, Any]]


def render(template: str, params: Dict[str, Any]) -> str:
    text = template.format(**params).replace("\n", " ").strip()
    return " ".join(text.split())


# -----------------------------
# Utilities
# -----------------------------

def rint(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)

def rfloat(rng: random.Random, lo: float, hi: float, nd: int = 3) -> float:
    return round(rng.uniform(lo, hi), nd)

def tries(sampler, max_tries: int = 120_000) -> Dict[str, Any]:
    last = None
    for _ in range(max_tries):
        try:
            return sampler()
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to sample valid instance. Last error: {last}")

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Number theory helpers
# -----------------------------

def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def inv_mod(a: int, m: int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("no inverse")
    return x % m

def crt_pair(a1: int, m1: int, a2: int, m2: int) -> Tuple[int, int]:
    # assumes gcd(m1,m2)=1
    t = (a2 - a1) % m2
    inv = inv_mod(m1 % m2, m2)
    k = (t * inv) % m2
    x = a1 + k * m1
    M = m1 * m2
    return (x % M, M)

def crt3(a1: int, m1: int, a2: int, m2: int, a3: int, m3: int) -> int:
    x12, M12 = crt_pair(a1, m1, a2, m2)
    x, M = crt_pair(x12, M12, a3, m3)
    return x

def digit_sum(n: int) -> int:
    return sum(int(ch) for ch in str(abs(n)))

def lcm(a: int, b: int) -> int:
    return abs(a // math.gcd(a, b) * b)

def count_valid_f43(A: int, B: int, m: int, d1: int, d2: int, d3: int, r: int, k: int) -> int:
    cnt = 0
    for n in range(A, B + 1):
        if n % m != 0:
            continue
        if n % d1 == 0 or n % d2 == 0 or n % d3 == 0:
            continue
        if n % k != r % k:
            continue
        cnt += 1
    return cnt

def smallest_N_base_digitsum_divisible(base: int, L: int, S: int, m: int) -> Optional[int]:
    # Find smallest integer with exactly L digits in base, digit sum S, and N % m == 0.
    # Search digits in lexicographic order => smallest numeric for fixed L.
    if L <= 0 or base <= 1 or m <= 0:
        return None
    if S < 1 or S > (base - 1) * L:
        return None

    digits: List[int] = [0] * L

    def dfs(pos: int, sum_rem: int, mod_rem: int) -> bool:
        if pos == L:
            return (sum_rem == 0) and (mod_rem % m == 0)
        # prune sum
        max_possible = (base - 1) * (L - pos)
        if sum_rem < 0 or sum_rem > max_possible:
            return False
        min_d = 1 if pos == 0 else 0
        max_d = min(base - 1, sum_rem)
        for d in range(min_d, max_d + 1):
            # more pruning: remaining sum must be feasible
            rem = sum_rem - d
            if rem < 0:
                continue
            if rem > (base - 1) * (L - pos - 1):
                continue
            digits[pos] = d
            new_mod = (mod_rem * base + d) % m
            if dfs(pos + 1, rem, new_mod):
                return True
        return False

    ok = dfs(0, S, 0)
    if not ok:
        return None

    # construct integer value
    val = 0
    for d in digits:
        val = val * base + d
    return val

def diophantine_solutions_in_box(A: int, B: int, C: int, xmin: int, xmax: int, ymin: int, ymax: int) -> List[Tuple[int, int]]:
    sols = []
    for x in range(xmin, xmax + 1):
        rhs = C - A * x
        if B == 0:
            continue
        if rhs % B != 0:
            continue
        y = rhs // B
        if ymin <= y <= ymax:
            sols.append((x, y))
    return sols


# -----------------------------
# Finance helpers
# -----------------------------

def amortize_with_refi(
    P0: float,
    r1: float, M1: int,
    r2: float,
    pay: float,
    k_extra: int, extra: float,
    Bth: float,
    r3: float, fee: float, pay2: float,
    max_months: int = 600
) -> Tuple[int, float, bool, int]:
    """
    Simulate:
      - Monthly compounding.
      - Rate is r1 for months 1..M1, then r2 until refinance triggers.
      - Each month: interest accrues, then payment applied (min(balance+interest, payment)).
      - Every k_extra months: either extra payment, OR if balance < Bth and not refinanced: refinance:
           add fee to balance, switch rate to r3 and payment to pay2 (extra payment skipped that month).
    Returns: (payoff_month, total_interest_paid, refinanced?, refinance_month)
    """
    bal = float(P0)
    total_interest = 0.0
    refinanced = False
    refi_month = 0

    for month in range(1, max_months + 1):
        if bal <= 1e-8:
            return month - 1, round(total_interest, 2), refinanced, refi_month

        apr = r1 if month <= M1 else (r3 if refinanced else r2)
        rate = apr / 100.0 / 12.0

        interest = bal * rate
        total_interest += interest
        bal += interest

        pmt = pay2 if refinanced else pay
        pmt_eff = min(bal, pmt)
        bal -= pmt_eff

        if bal <= 1e-8:
            return month, round(total_interest, 2), refinanced, refi_month

        if (month % k_extra) == 0:
            if (not refinanced) and (bal < Bth):
                refinanced = True
                refi_month = month
                bal += fee
                # no extra payment on refinance month
            else:
                ex = min(bal, extra)
                bal -= ex

    raise ValueError("did not pay off within max_months")

def tiered_commission(n1: int, n2: int, p1: float, p2: float, c: float, c2: float, Q: int) -> Tuple[float, float]:
    # returns (phase1_commission, total_commission) based on unit-count quota, rates in percent
    phase1 = 0.0
    total = 0.0
    sold = 0
    # phase 1
    for _ in range(n1):
        sold += 1
        rate = c2 if sold > Q else c
        phase1 += (rate / 100.0) * p1
        total += (rate / 100.0) * p1
    # phase 2
    for _ in range(n2):
        sold += 1
        rate = c2 if sold > Q else c
        total += (rate / 100.0) * p2
    return (phase1, total)


# -----------------------------
# 41) CRT + digit sum branch
# -----------------------------

def sample_f41(rng: random.Random) -> Dict[str, Any]:
    def _one():
        # pairwise coprime moduli
        mods = []
        while len(mods) < 3:
            m = rint(rng, 5, 45)
            if any(math.gcd(m, mm) != 1 for mm in mods):
                continue
            mods.append(m)
        m1, m2, m3 = mods

        a1 = rint(rng, 0, m1 - 1)
        a2 = rint(rng, 0, m2 - 1)
        a3 = rint(rng, 0, m3 - 1)

        n0 = crt3(a1, m1, a2, m2, a3, m3)
        if n0 == 0:
            n0 = m1 * m2 * m3  # smallest positive
        ds = digit_sum(n0)

        # force branch randomly by choosing S around ds
        want_ge = (rng.random() < 0.5)
        if want_ge:
            S = rint(rng, max(1, ds - 8), ds)
        else:
            S = rint(rng, ds + 1, ds + 10)

        # choose k so both n0+k and n0-k positive (especially for subtract branch)
        if n0 <= 5:
            raise ValueError("n too small")
        k = rint(rng, 1, min(50, n0 - 1))

        # ensure subtract branch would still be positive if chosen
        if n0 - k <= 0:
            raise ValueError("adjustment not positive")

        return {"a1": a1, "m1": m1, "a2": a2, "m2": m2, "a3": a3, "m3": m3, "S": S, "k": k}
    return tries(_one)


# -----------------------------
# 42) gcd/lcm + sum + parity-branch constraint (case given)
# -----------------------------

def sample_f42(rng: random.Random) -> Dict[str, Any]:
    def _one():
        case_even = (rng.random() < 0.5)

        g = rint(rng, 2, 30)

        # choose coprime a,b and enforce a>b (so x>y)
        a = rint(rng, 2, 40)
        b = rint(rng, 2, 40)
        if math.gcd(a, b) != 1:
            raise ValueError("a,b not coprime")
        if a == b:
            raise ValueError("avoid equal")
        if a < b:
            a, b = b, a  # ensure a>b

        x = g * a
        y = g * b
        if x == y:
            raise ValueError("avoid equal")

        # enforce desired parity of x
        if case_even:
            if x % 2 != 0:
                raise ValueError("x not even")
        else:
            if x % 2 == 0:
                raise ValueError("x not odd")

        L = lcm(x, y)  # with gcd(a,b)=1, this is g*a*b

        if case_even:
            Ssum = x + y
            return {
                "g": g,
                "L": L,
                "parity_case": "even",
                "extra_constraint": f"x+y={Ssum}",
            }
        else:
            D = x - y  # positive since x>y
            return {
                "g": g,
                "L": L,
                "parity_case": "odd",
                "extra_constraint": f"x−y={D}",
            }

    return tries(_one)


# -----------------------------
# 43) count in interval with exclusions + congruence + endpoint adjustment
# -----------------------------

def sample_f43(rng: random.Random) -> Dict[str, Any]:
    def _one():
        A = rint(rng, 1, 1500)
        B = rint(rng, A + 200, A + 2200)

        m = rint(rng, 2, 40)
        k = rint(rng, 2, 40)
        r = rint(rng, 0, k - 1)

        # choose d's that do NOT divide m and are distinct
        dset = set()
        while len(dset) < 3:
            d = rint(rng, 2, 45)
            if d == m or d == k:
                continue
            if (m % d) == 0:
                continue
            dset.add(d)
        d1, d2, d3 = sorted(dset)

        cnt = count_valid_f43(A, B, m, d1, d2, d3, r, k)
        if cnt == 0:
            raise ValueError("trivial zero count")

        # choose threshold T near cnt so post-processing is meaningful
        Tthr = rint(rng, max(0, cnt - 3), cnt + 3)
        s = rint(rng, 1, 10)

        return {"A": A, "B": B, "m": m, "d1": d1, "d2": d2, "d3": d3, "r": r, "k": k, "T": Tthr, "s": s}
    return tries(_one)


# -----------------------------
# 44) base representation constraints + conditional base switch
# -----------------------------

def sample_f44(rng: random.Random) -> Dict[str, Any]:
    def _one():
        fallback = (rng.random() < 0.5)

        L = rint(rng, 3, 6)
        b = rint(rng, 2, 12)
        b2 = rint(rng, 2, 12)
        if b2 == b:
            raise ValueError("bases must differ")

        m = rint(rng, 2, 60)

        # choose S feasible in at least one base
        # pick S not too extreme to keep search manageable
        S = rint(rng, 2, 3 * L + 10)

        N1 = smallest_N_base_digitsum_divisible(b, L, S, m)
        N2 = smallest_N_base_digitsum_divisible(b2, L, S, m)

        if fallback:
            # need no solution in base b, but solution in base b2
            if N1 is not None:
                raise ValueError("base b already has solution")
            if N2 is None:
                raise ValueError("base b2 also has no solution")
        else:
            # need solution in base b
            if N1 is None:
                raise ValueError("base b has no solution")

        return {"b": b, "L": L, "S": S, "m": m, "b2": b2}
    return tries(_one)


# -----------------------------
# 45) bounded linear Diophantine with objective + tie-break
# -----------------------------

def sample_f45(rng: random.Random) -> Dict[str, Any]:
    def _one():
        A = rint(rng, 2, 30)
        B = rint(rng, 2, 30)
        # choose a seed solution
        x0 = rint(rng, -25, 25)
        y0 = rint(rng, -25, 25)
        C = A * x0 + B * y0

        # bounds around the seed, sometimes allowing multiple solutions
        xmin = x0 - rint(rng, 5, 30)
        xmax = x0 + rint(rng, 5, 30)
        ymin = y0 - rint(rng, 5, 30)
        ymax = y0 + rint(rng, 5, 30)
        if xmin > xmax or ymin > ymax:
            raise ValueError("bad bounds")

        w1 = rint(rng, 1, 8)
        w2 = rint(rng, 1, 8)

        sols = diophantine_solutions_in_box(A, B, C, xmin, xmax, ymin, ymax)
        if not sols:
            raise ValueError("no feasible solutions (should not happen)")

        # encourage nontrivial cases sometimes
        if len(sols) == 1 and rng.random() < 0.6:
            raise ValueError("too unique too often")

        return {
            "Acoef": A, "Bcoef": B, "Ccoef": C,
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "w1": w1, "w2": w2
        }
    return tries(_one)


# -----------------------------
# 46) loan with variable APR, extra payments, and refinance trigger
# -----------------------------

def sample_f46(rng: random.Random) -> Dict[str, Any]:
    def _one():
        want_refi = (rng.random() < 0.5)

        P0 = float(rint(rng, 5_000, 200_000))
        r1 = rfloat(rng, 2.0, 14.0, 3)
        r2 = rfloat(rng, 3.0, 18.0, 3)
        M1 = rint(rng, 6, 48)

        k_extra = rint(rng, 3, 12)
        extra = float(rint(rng, 50, 2000))

        # choose payment using an amortization-like formula for a target term
        term_guess = rint(rng, 24, 180)
        rr = (r1 / 100.0) / 12.0
        if rr <= 0:
            raise ValueError("bad r1")
        pay = P0 * rr / (1.0 - (1.0 + rr) ** (-term_guess))
        pay *= rng.uniform(0.95, 1.25)
        pay = round(pay, 2)

        # must exceed first-month interest
        if pay <= round(P0 * rr + 1.0, 2):
            raise ValueError("payment too small to amortize")

        # refinance settings
        r3 = rfloat(rng, 1.5, 12.0, 3)
        fee = float(rint(rng, 200, 4500))
        pay2 = round(pay * rng.uniform(0.85, 1.25), 2)
        if pay2 <= round(P0 * (r3 / 100.0) / 12.0 + 1.0, 2):
            raise ValueError("pay2 too small")

        # set Bth to control refi likelihood by probing balances without refi
        # Use a trial simulation with very small Bth to effectively disable refinance.
        base_months, _, _, _ = amortize_with_refi(
            P0, r1, M1, r2, pay, k_extra, extra,
            Bth=1e-9, r3=r3, fee=fee, pay2=pay2, max_months=600
        )
        # collect balances at multiples of k by simulating again but capturing internal states (approx via repeated runs)
        # Instead: pick Bth from a reasonable range and validate refi behavior via simulation.
        if want_refi:
            Bth = float(rint(rng, int(0.10 * P0), int(0.70 * P0)))
        else:
            Bth = float(rint(rng, 1, int(0.05 * P0)))

        payoff_m, tot_int, did_refi, refi_m = amortize_with_refi(
            P0, r1, M1, r2, pay, k_extra, extra,
            Bth=Bth, r3=r3, fee=fee, pay2=pay2, max_months=600
        )

        if payoff_m <= 0 or payoff_m > 600:
            raise ValueError("bad payoff horizon")

        if want_refi and (not did_refi):
            raise ValueError("wanted refi but none occurred")
        if (not want_refi) and did_refi:
            raise ValueError("did not want refi but it occurred")

        # keep runtime reasonable by avoiding extremely long loans
        if payoff_m > 420 and rng.random() < 0.8:
            raise ValueError("too long too often")

        return {
            "P0": int(P0), "r1": r1, "M1": M1, "r2": r2,
            "pay": pay, "k": k_extra, "extra": int(extra),
            "Bth": int(Bth), "r3": r3, "F": int(fee), "pay2": pay2
        }
    return tries(_one)


# -----------------------------
# 47) commission + quota bonus + clawback with returns
# -----------------------------

def sample_f47(rng: random.Random) -> Dict[str, Any]:
    def _one():
        n1 = rint(rng, 10, 140)
        n2 = rint(rng, 10, 160)
        p1 = rfloat(rng, 10.0, 350.0, 2)
        p2 = rfloat(rng, 10.0, 400.0, 2)

        c = rfloat(rng, 1.0, 12.0, 2)
        c2 = rfloat(rng, c + 0.5, min(30.0, c + 18.0), 2)

        Q = rint(rng, 5, n1 + n2 - 5)

        Rth = rint(rng, 0, max(0, n2 - 1))
        # choose realized returns explicitly (needed for determinism)
        ret = rint(rng, 0, n2)
        # enforce both regimes across instances by resampling sometimes
        if (ret <= Rth) and rng.random() < 0.45:
            raise ValueError("prefer crossing sometimes")
        if (ret > Rth) and rng.random() < 0.45:
            raise ValueError("prefer non-crossing sometimes")

        cb = rfloat(rng, 5.0, 60.0, 2)
        t = rfloat(rng, 5.0, 45.0, 2)

        comm1, comm_total = tiered_commission(n1, n2, p1, p2, c, c2, Q)
        after_tax = comm_total * (1.0 - t / 100.0)
        claw = (cb / 100.0) * comm1 if (ret > Rth) else 0.0
        take_home = after_tax - claw
        if take_home <= 0:
            raise ValueError("negative take-home")

        return {
            "n1": n1, "n2": n2, "p1": p1, "p2": p2,
            "c": c, "c2": c2, "Q": Q,
            "ret": ret, "Rth": Rth, "cb": cb, "t": t
        }
    return tries(_one)


# -----------------------------
# 48) recipe scaling with loss + rounding + conditional discount
# -----------------------------

def sample_f48(rng: random.Random) -> Dict[str, Any]:
    def _one():
        want_discount = (rng.random() < 0.5)

        a = rint(rng, 50, 600)       # grams X
        b = rint(rng, 30, 800)       # mL Y
        c = rint(rng, 30, 600)       # grams Z
        N = rint(rng, 2, 14)         # base servings
        Sserv = rint(rng, N + 1, N + 25)

        loss = rfloat(rng, 2.0, 25.0, 2)
        step = rint(rng, 5, 50)      # mL step
        d = rfloat(rng, 3.0, 40.0, 2)

        cx = rfloat(rng, 0.01, 0.15, 3)  # cost per gram
        cy = rfloat(rng, 0.001, 0.10, 3) # cost per mL
        cz = rfloat(rng, 0.01, 0.20, 3)  # cost per gram

        scale = Sserv / N
        x_need = a * scale / (1.0 - loss / 100.0)
        y_need = b * scale
        # ensure rounding matters
        if abs((y_need / step) - round(y_need / step)) < 1e-9:
            raise ValueError("rounding would not matter")

        y_round = math.ceil(y_need / step) * step
        z_need = c * scale

        pre_cost = cx * x_need + cy * y_round + cz * z_need
        pre_cost = round(pre_cost, 2)

        if want_discount:
            Cth = round(pre_cost - rfloat(rng, 0.50, min(20.0, max(0.55, pre_cost * 0.30))), 2)
        else:
            Cth = round(pre_cost + rfloat(rng, 0.50, min(20.0, max(0.55, pre_cost * 0.30))), 2)

        # sanity
        if Cth <= 0:
            raise ValueError("Cth must be positive")

        return {
            "a": a, "b": b, "c": c, "N": N, "S": Sserv,
            "loss": loss, "step": step,
            "Cth": Cth, "d": d,
            "cx": cx, "cy": cy, "cz": cz
        }
    return tries(_one)


# -----------------------------
# 49) shipping with volumetric weight + tiered rates + conditional fee/surcharge interaction
# -----------------------------

def sample_f49(rng: random.Random) -> Dict[str, Any]:
    def _one():
        # zones
        zones = ["EU", "NA", "APAC", "LATAM", "ME"]
        rng.shuffle(zones)
        ZONE_SET = "{" + ",".join(zones[:rint(rng, 2, 4)]) + "}"
        zone = zones[rint(rng, 0, len(zones) - 1)]
        in_zone = zone in ZONE_SET

        want_volumetric = (rng.random() < 0.5)

        div = choice = 5000  # typical volumetric divisor
        # actual weight
        Wkg = rfloat(rng, 1.0, 30.0, 2)

        if want_volumetric:
            # choose dimensions so volumetric > actual
            Lcm = rint(rng, 35, 120)
            Wcm = rint(rng, 25, 90)
            Hcm = rint(rng, 15, 80)
        else:
            # smallish dims
            Lcm = rint(rng, 10, 55)
            Wcm = rint(rng, 10, 45)
            Hcm = rint(rng, 5, 40)

        vol_wt = (Lcm * Wcm * Hcm) / div
        bill = max(Wkg, vol_wt)
        # enforce volumetric dominance condition as desired
        if want_volumetric and not (vol_wt > Wkg + 0.5):
            raise ValueError("volumetric not dominating enough")
        if (not want_volumetric) and not (Wkg >= vol_wt):
            raise ValueError("actual should dominate")

        K = rfloat(rng, 5.0, 25.0, 2)
        r1 = rfloat(rng, 0.5, 8.0, 2)
        r2 = rfloat(rng, max(r1 + 0.2, 0.7), 12.0, 2)
        fs = rfloat(rng, 2.0, 30.0, 2)

        F = rfloat(rng, 2.0, 40.0, 2)
        Deltafs = rfloat(rng, 1.0, 20.0, 2)

        # choose T to sometimes trigger waiver when in_zone
        if in_zone:
            if rng.random() < 0.5:
                # waiver triggers: bill > T
                T = rfloat(rng, max(0.5, bill - 6.0), bill - 0.2, 2)
                if not (bill > T):
                    raise ValueError("waiver should trigger")
            else:
                # waiver does not trigger: bill <= T
                T = rfloat(rng, bill + 0.2, bill + 8.0, 2)
                if not (bill <= T):
                    raise ValueError("waiver should not trigger")
        else:
            # T still defined
            T = rfloat(rng, 5.0, 35.0, 2)

        return {
            "Wkg": Wkg, "Lcm": Lcm, "Wcm": Wcm, "Hcm": Hcm, "div": div,
            "r1": r1, "K": K, "r2": r2,
            "fs": fs,
            "ZONE_SET": ZONE_SET, "zone": zone,
            "F": F, "T": T, "Deltafs": Deltafs
        }
    return tries(_one)


# -----------------------------
# 50) multi-phase production with scrap threshold -> inspection regime
# -----------------------------

def sample_f50(rng: random.Random) -> Dict[str, Any]:
    def _one():
        N = rint(rng, 50, 800)
        D1 = rint(rng, 5, 40)
        p1 = rfloat(rng, 0.03, 0.35, 3)

        R = rint(rng, max(1, int(0.15 * N)), max(2, int(0.80 * N)))
        s = rfloat(rng, 0.40, 0.95, 3)

        # ensure rework capacity is relevant: expected defects sometimes exceed R
        if N * p1 <= R + 1e-6:
            if rng.random() < 0.75:
                raise ValueError("rework capacity not binding enough")

        N2 = rint(rng, int(1.05 * N), int(2.0 * N))
        D2 = rint(rng, 5, 40)

        p2 = rfloat(rng, 0.03, 0.35, 3)
        p3 = rfloat(rng, 0.01, min(0.30, p2 - 0.005), 3)  # inspection reduces defects

        # expected scrap after phase 1 (approx)
        exp_def = N * p1
        exp_rework = min(R, exp_def)
        exp_scrap_day = exp_rework * (1.0 - s) + max(0.0, exp_def - R)
        exp_scrap_total = exp_scrap_day * D1

        # choose threshold around expected scrap to make both regimes plausible
        Sth = int(max(1.0, exp_scrap_total * rng.uniform(0.6, 1.4)))
        C = rint(rng, 50, 5000)  # per-day inspection cost

        # avoid extremely tiny expected scrap
        if exp_scrap_total < 2.0:
            raise ValueError("scrap too small")

        return {
            "N": N, "D1": D1, "p1": p1,
            "R": R, "s": s,
            "D2": D2, "N2": N2, "p2": p2, "p3": p3,
            "Sth": Sth, "C": C
        }
    return tries(_one)


# -----------------------------
# Families 41–50
# -----------------------------

FAMILIES: List[ProblemFamily] = [
    ProblemFamily(
        "f41",
        "CRT with range constraint and digit-sum branch",
        "Find the smallest positive integer n such that n ≡ {a1} (mod {m1}), n ≡ {a2} (mod {m2}), and n ≡ {a3} (mod {m3}). Additionally, if the digit sum of n is at least {S}, report n+{k}; otherwise report n-{k}.",
        sample_f41,
    ),
    ProblemFamily(
        "f42",
        "GCD/LCM with branch-specific linear constraint and parity condition",
        "Positive integers x and y satisfy gcd(x,y)={g} and lcm(x,y)={L}. "
        "Assume x>y. You are told that x is {parity_case}, and additionally {extra_constraint}. "
        "Find (x,y).",
        sample_f42,
    ),
    ProblemFamily(
        "f43",
        "Counting integers under multiple modular exclusions with endpoint adjustment",
        "How many integers n in the interval [{A},{B}] are divisible by {m} but not divisible by any of {d1}, {d2}, {d3}, and also satisfy n ≡ {r} (mod {k})? If the count exceeds {T}, subtract {s}; otherwise add {s}.",
        sample_f43,
    ),
    ProblemFamily(
        "f44",
        "Base representation with digit constraints and conditional base switch",
        "A number N has a base-{b} representation with exactly {L} digits, digit sum {S}, and N is divisible by {m}. If no such N exists in base {b}, switch to base {b2} while keeping the same L and S. Find the smallest valid N.",
        sample_f44,
    ),
    ProblemFamily(
        "f45",
        "Diophantine with bounded solutions and tie-breaker rule",
        "Find integers (x,y) satisfying {Acoef}x+{Bcoef}y={Ccoef}, with bounds {xmin}≤x≤{xmax}, {ymin}≤y≤{ymax}. If multiple solutions exist, choose the one minimizing {w1}|x|+{w2}|y|; if still tied, choose the one with larger x. Report the chosen (x,y).",
        sample_f45,
    ),
    ProblemFamily(
        "f46",
        "Loan with variable APR, extra payments, and refinance trigger",
        "A loan has principal {P0} with APR {r1}% compounded monthly for the first {M1} months, then APR becomes {r2}%. Monthly payment is {pay}. Additionally, every {k} months you make an extra payment {extra} unless the remaining balance is below {Bth}, in which case you refinance to APR {r3}% with one-time fee {F} and new monthly payment {pay2}. Compute total interest paid and the payoff month.",
        sample_f46,
    ),
    ProblemFamily(
        "f47",
        "Commission + quota bonus + clawback with returns",
        "A salesperson sells {n1} units in Phase 1 and {n2} units in Phase 2. Unit price is {p1} then {p2}. Commission is {c}% but becomes {c2}% for units sold beyond quota {Q} (counting units across both phases). Phase-2 returns are {ret} units; if returns exceed {Rth} units, a clawback of {cb}% of Phase-1 commission applies. Taxes are {t}% applied after the higher-rate quota bonus is accounted for, but before clawback. Compute final take-home pay (commission after tax and any clawback).",
        sample_f47,
    ),
    ProblemFamily(
        "f48",
        "Multi-stage recipe scaling with yield loss and rounding",
        "A recipe uses {a} grams ingredient X, {b} mL ingredient Y, and {c} grams ingredient Z to produce {N} servings. When scaling to {S} servings, ingredient X experiences {loss}% yield loss during cooking (so you must start with more), and ingredient Y must be measured in multiples of {step} mL (rounded up). If the pre-discount total cost exceeds {Cth}, a discount {d}% applies only to ingredient Z. Given unit costs cx={cx} per gram, cy={cy} per mL, cz={cz} per gram, compute total cost and final amounts of each ingredient used.",
        sample_f48,
    ),
    ProblemFamily(
        "f49",
        "Shipping with volumetric weight, tiered rates, and surcharge interaction",
        "A package has actual weight {Wkg} kg and dimensions {Lcm}×{Wcm}×{Hcm} cm. Volumetric weight is (L·W·H)/{div} kg, and billable weight is max(actual, volumetric). Rate is {r1} per kg up to {K} kg and {r2} beyond, plus fuel surcharge {fs}%. Destination zone is {zone}. If destination is in {ZONE_SET}, add flat {F} unless billable weight exceeds {T}, in which case the flat fee is waived but the fuel surcharge increases by {Deltafs}%. Compute total charge.",
        sample_f49,
    ),
    ProblemFamily(
        "f50",
        "Multi-phase production with scrap, rework capacity, and final quality constraint",
        "A factory produces {N} items per day for {D1} days with defect rate {p1}. Defectives can be reworked at capacity {R} items/day with success probability {s}; unreworked defectives are scrapped and failed reworks are scrapped. For the next {D2} days, production increases to {N2} but defect rate becomes {p2} unless the cumulative scrap so far exceeds {Sth}, in which case defect rate reduces to {p3} due to extra inspection cost {C} per day. Find the expected number of sellable items and the expected total inspection cost.",
        sample_f50,
    ),
]


# -----------------------------
# Generation + outputs
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


def main(out_dir: str = "out_nt_fin", n_per_family: int = 10, seed: int = 12345) -> None:
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
