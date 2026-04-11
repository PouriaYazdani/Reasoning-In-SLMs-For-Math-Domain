from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple
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


def rprob(rng: random.Random, lo: float = 0.05, hi: float = 0.95, nd: int = 3) -> float:
    return rfloat(rng, lo, hi, nd)


def choice(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def tries(sampler, max_tries: int = 80_000) -> Dict[str, Any]:
    last = None
    for _ in range(max_tries):
        try:
            return sampler()
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to sample valid instance. Last error: {last}")


def set_str(ints: List[int]) -> str:
    return "{" + ",".join(map(str, sorted(ints))) + "}"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Family 23: precedence + uncertainty + threshold optimization
# -----------------------------


def project_makespan(dA: int, dB: int, dC: int, dD: int) -> int:
    # A then max(B,C) then D
    return dA + max(dB, dC) + dD


def min_cost_one_task_overtime(
    dA: int,
    dB: int,
    dC: int,
    dD: int,
    r_pct: int,
    c_per_hr: int,
    T: int,
    P_per_hr: int,
    T2: int,
    B_bonus: int,
) -> Tuple[int, str, int, int]:
    """
    Choose at most one task to reduce by integer hours up to floor(r% * duration).
    Returns (min_cost, task_name, reduction_hours, makespan).
    """
    tasks = [("none", 0), ("A", dA), ("B", dB), ("C", dC), ("D", dD)]
    best = None

    for name, dur in tasks:
        max_red = 0 if name == "none" else int(math.floor(r_pct / 100.0 * dur))
        for red in range(0, max_red + 1):
            a = dA - (red if name == "A" else 0)
            b = dB - (red if name == "B" else 0)
            c = dC - (red if name == "C" else 0)
            d = dD - (red if name == "D" else 0)
            if min(a, b, c, d) <= 0:
                continue

            ms = project_makespan(a, b, c, d)
            overtime_cost = c_per_hr * red
            penalty = 0 if ms <= T else P_per_hr * (ms - T)
            bonus = -B_bonus if ms <= T2 else 0
            total = overtime_cost + penalty + bonus

            cand = (total, name, red, ms)
            if best is None or cand[0] < best[0]:
                best = cand

    if best is None:
        raise ValueError("No feasible overtime choice.")
    return best


def sample_f23(rng: random.Random) -> Dict[str, Any]:
    def _one():
        dA = rint(rng, 2, 14)
        dB = rint(rng, 2, 16)
        dC = rint(rng, 2, 16)
        dD = rint(rng, 2, 14)

        Delta = rint(rng, 1, 10)
        p = rprob(rng, 0.15, 0.85, 3)

        r_pct = rint(rng, 10, 45)
        c_per_hr = rint(rng, 10, 120)
        P_per_hr = rint(rng, 30, 300)
        B_bonus = rint(rng, 100, 2000)

        # base makespans
        ms0 = project_makespan(dA, dB, dC, dD)
        ms1 = project_makespan(dA, dB, dC + Delta, dD)

        # thresholds around baseline to avoid trivial always/never
        T = rint(rng, min(ms0, ms1) - 2, max(ms0, ms1) + 3)
        T = max(T, 3)
        T2 = rint(rng, max(2, T - 6), min(T, T - 1))  # ensure T2 < T

        # compute optimal decisions after observing scenario
        best0 = min_cost_one_task_overtime(
            dA, dB, dC, dD, r_pct, c_per_hr, T, P_per_hr, T2, B_bonus
        )
        best1 = min_cost_one_task_overtime(
            dA, dB, dC + Delta, dD, r_pct, c_per_hr, T, P_per_hr, T2, B_bonus
        )

        # avoid domination: force that at least one scenario uses overtime or differs
        uses_ot0 = best0[2] > 0
        uses_ot1 = best1[2] > 0
        if not (uses_ot0 or uses_ot1):
            raise ValueError("never overtime")

        # make sure not always the same decision with same reduction
        if best0[1] == best1[1] and best0[2] == best1[2]:
            raise ValueError("decision identical; resample")

        return {
            "dA": dA,
            "dB": dB,
            "dC": dC,
            "dD": dD,
            "r": r_pct,
            "c": c_per_hr,
            "T": T,
            "P": P_per_hr,
            "T2": T2,
            "B": B_bonus,
            "Delta": Delta,
            "p": p,
        }

    return tries(_one)


# -----------------------------
# Family 24: bin packing with incompatibilities + tiered shipping + surcharge
# -----------------------------


def greedy_feasible_packing(
    weights: List[int], Cap: int, fragile: set, heavy: set
) -> bool:
    # Very simple feasibility checker:
    # - No fragile item can share with heavy item.
    # We'll pack sequentially into boxes; if cannot, open new.
    boxes: List[Dict[str, Any]] = (
        []
    )  # {"w": int, "has_fragile": bool, "has_heavy": bool}

    for idx, w in enumerate(weights, start=1):
        if w > Cap:
            return False
        is_f = idx in fragile
        is_h = idx in heavy
        placed = False
        for box in boxes:
            if box["w"] + w > Cap:
                continue
            if (is_f and box["has_heavy"]) or (is_h and box["has_fragile"]):
                continue
            box["w"] += w
            box["has_fragile"] = box["has_fragile"] or is_f
            box["has_heavy"] = box["has_heavy"] or is_h
            placed = True
            break
        if not placed:
            boxes.append({"w": w, "has_fragile": is_f, "has_heavy": is_h})
    return True


def sample_f24(rng: random.Random) -> Dict[str, Any]:
    def _one():
        N = rint(rng, 6, 10)
        Cap = rint(rng, 18, 40)

        # weights: ensure not all fit in one box, not all huge
        weights = [rint(rng, 2, min(18, Cap)) for _ in range(N)]
        if sum(weights) <= Cap:
            raise ValueError("trivial one-box case")

        # make incompatible sets non-empty and disjoint
        items = list(range(1, N + 1))
        rng.shuffle(items)
        fragile_sz = rint(rng, 1, max(1, N // 3))
        heavy_sz = rint(rng, 1, max(1, N // 3))
        fragile = set(items[:fragile_sz])
        heavy = set(items[fragile_sz : fragile_sz + heavy_sz])
        if fragile & heavy:
            raise ValueError("overlap sets")

        # set heavy items to be heavier on average
        for i in heavy:
            weights[i - 1] = rint(rng, max(6, Cap // 3), min(Cap, max(6, Cap - 1)))
        for i in fragile:
            weights[i - 1] = rint(rng, 2, min(10, Cap))

        if not greedy_feasible_packing(weights, Cap, fragile, heavy):
            raise ValueError("packing infeasible")

        # tiered shipping
        m = rint(rng, 1, 4)
        c1 = rint(rng, 20, 120)
        c2 = rint(rng, 30, 160)
        if c2 < c1:
            # allow but not always; resample sometimes
            if rng.random() < 0.6:
                raise ValueError("want tier effect")

        theta = rint(rng, max(1, int(0.60 * Cap)), Cap - 1)
        S = rint(rng, 10, 120)

        return {
            "N": N,
            "weights": "[" + ",".join(map(str, weights)) + "]",
            "Cap": Cap,
            "c1": c1,
            "m": m,
            "c2": c2,
            "S": S,
            "theta": theta,
            "FRAGILE_SET": set_str(list(fragile)),
            "HEAVY_SET": set_str(list(heavy)),
        }

    return tries(_one)


# -----------------------------
# Family 25: shortest path with toll-discount state + time-dependent edge availability
# -----------------------------


def sample_f25(rng: random.Random) -> Dict[str, Any]:
    def _one():
        Start = "S"
        End = "T"
        X = "X"
        nodes = ["S", "A", "B", "X", "T"]

        # Construct edges with intended structure:
        # - Fast to X directly: S->X (arrive early)
        # - Slower to X via B: S->B->X (arrive late)
        # - e* is X->T which becomes unavailable if you enter X before tau
        # - backup from X: X->A->T
        edges = [
            ("S", "X", rint(rng, 2, 5), rint(rng, 6, 14)),  # time, toll
            ("S", "B", rint(rng, 4, 8), rint(rng, 1, 7)),
            ("B", "X", rint(rng, 4, 8), rint(rng, 1, 7)),
            ("X", "T", rint(rng, 3, 6), rint(rng, 6, 16)),  # e*
            ("X", "A", rint(rng, 2, 6), rint(rng, 1, 8)),
            ("A", "T", rint(rng, 3, 7), rint(rng, 1, 8)),
            ("S", "A", rint(rng, 5, 10), rint(rng, 1, 8)),
            ("B", "T", rint(rng, 6, 12), rint(rng, 1, 10)),
        ]

        t_SX = next(t for (u, v, t, p) in edges if u == "S" and v == "X")
        t_SBX = next(t for (u, v, t, p) in edges if u == "S" and v == "B") + next(
            t for (u, v, t, p) in edges if u == "B" and v == "X"
        )

        if not (t_SX < t_SBX):
            raise ValueError("need direct-to-X faster than via B")

        # choose tau strictly between them so entering X early blocks e*
        tau = rint(rng, t_SX + 1, t_SBX - 1)

        TollCap = rint(rng, 8, 25)
        d = rint(rng, 10, 50)  # discount %
        alpha = rint(rng, 1, 8)
        beta = rint(rng, 1, 8)

        # ensure at least 2 plausible routes exist regardless of e*:
        # S->A->T and S->B->T must exist (they do by construction)

        e_star = "X->T"

        edges_str = "; ".join(
            [f"{u}->{v}(time={t}, toll={p})" for (u, v, t, p) in edges]
        )

        return {
            "Start": Start,
            "End": End,
            "X": X,
            "EDGES": edges_str,
            "TollCap": TollCap,
            "d": d,
            "tau": tau,
            "e_star": e_star,
            "alpha": alpha,
            "beta": beta,
        }

    return tries(_one)


# -----------------------------
# Family 26: inventory with lead time + forced emergency when below S
# -----------------------------


def simulate_no_orders(I0: int, H: int, H1: int, d1: int, d2: int) -> List[int]:
    inv = I0
    hist = []
    for day in range(1, H + 1):
        dem = d1 if day <= H1 else d2
        inv -= dem
        hist.append(inv)
    return hist


def simulate_with_single_order_day1(
    I0: int, H: int, H1: int, d1: int, d2: int, L: int, Q: int
) -> List[int]:
    inv = I0
    hist = []
    for day in range(1, H + 1):
        if day == 1 + L:
            inv += Q
        dem = d1 if day <= H1 else d2
        inv -= dem
        hist.append(inv)
    return hist


def sample_f26(rng: random.Random) -> Dict[str, Any]:
    def _one():
        H = rint(rng, 10, 22)
        H1 = rint(rng, 3, H - 3)
        H1_plus_1 = H1 + 1

        d1 = rint(rng, 2, 12)
        d2 = rint(rng, 2, 14)

        L = rint(rng, 1, 4)
        Le = rint(rng, 1, 3)

        # choose S and I0 so you can survive L days above S if you plan,
        # but without orders you eventually drop below S.
        S = rint(rng, 8, 40)

        I0 = rint(rng, S + d1 * (L + 1), S + d1 * (L + 1) + 60)

        no_ord = simulate_no_orders(I0, H, H1, d1, d2)
        if min(no_ord) >= S:
            raise ValueError("emergency never triggers without orders")

        # check avoidability with one large order arriving after L days
        Q = rint(rng, 20, 200)
        with_ord = simulate_with_single_order_day1(I0, H, H1, d1, d2, L, Q)
        if min(with_ord) < S:
            # cannot keep above S with simple early order; resample
            raise ValueError("emergency unavoidable even with early large order")

        K = rint(rng, 30, 300)
        c = rint(rng, 2, 20)
        h = rint(rng, 1, 5)
        p = rint(rng, 10, 120)

        E = rint(rng, 20, 200)
        cp = rint(rng, c + 1, c + 25)

        return {
            "d1": d1,
            "H1": H1,
            "d2": d2,
            "H": H,
            "H1_plus_1": H1_plus_1,
            "I0": I0,
            "L": L,
            "K": K,
            "c": c,
            "h": h,
            "p": p,
            "S": S,
            "E": E,
            "Le": Le,
            "cp": cp,
        }

    return tries(_one)


# -----------------------------
# Family 27: inclusion-exclusion consistency
# -----------------------------


def sample_f27(rng: random.Random) -> Dict[str, Any]:
    def _one():
        # choose 8-region counts directly
        onlyM = rint(rng, 0, 40)
        onlyP = rint(rng, 0, 40)
        onlyC = rint(rng, 0, 40)
        MP_only = rint(rng, 0, 25)
        MC_only = rint(rng, 0, 25)
        PC_only = rint(rng, 0, 25)
        all3 = rint(rng, 0, 20)
        none = rint(rng, 0, 40)

        # ensure not trivial and union > 0
        union = onlyM + onlyP + onlyC + MP_only + MC_only + PC_only + all3
        if union == 0:
            raise ValueError("empty union")
        if min(onlyM, onlyP, onlyC) == 0 and rng.random() < 0.5:
            raise ValueError("too degenerate (often)")

        N = union + none
        a = onlyM + MP_only + MC_only + all3
        b = onlyP + MP_only + PC_only + all3
        c = onlyC + MC_only + PC_only + all3
        ab = MP_only + all3
        ac = MC_only + all3
        bc = PC_only + all3
        abc = all3
        z = none

        # consistency check via inclusion-exclusion
        union_ie = a + b + c - ab - ac - bc + abc
        if union_ie != union:
            raise ValueError("IE mismatch")

        # exact-one count must be nonnegative
        exact_one = onlyM + onlyP + onlyC
        if exact_one <= 0:
            raise ValueError("exactly-one trivial")

        return {
            "N": N,
            "a": a,
            "b": b,
            "c": c,
            "ab": ab,
            "ac": ac,
            "bc": bc,
            "abc": abc,
            "z": z,
        }

    return tries(_one)


# -----------------------------
# Family 28: parallel-edge flow with threshold upgrade (nonlinear)
# -----------------------------


def min_cost_parallel_flow(
    F: int, caps: List[int], costs: List[int], u: int, dcap: int, dcost: int
) -> Tuple[int, Tuple[int, ...]]:
    k = len(caps)
    best_cost = None
    best_alloc = None

    # brute force integer allocations for k=3
    if k != 3:
        raise ValueError("This helper assumes 3 parallel edges.")

    for f1 in range(0, F + 1):
        for f2 in range(0, F - f1 + 1):
            f3 = F - f1 - f2
            flows = [f1, f2, f3]
            ok = True
            total = 0
            for i in range(3):
                fi = flows[i]
                cap_i = caps[i] + (dcap if fi > u else 0)
                if fi > cap_i:
                    ok = False
                    break
                ci = costs[i] + (dcost if fi > u else 0)
                total += ci * fi
            if not ok:
                continue
            if best_cost is None or total < best_cost:
                best_cost = total
                best_alloc = tuple(flows)

    if best_cost is None:
        raise ValueError("No feasible flow allocation.")
    return best_cost, best_alloc


def sample_f28(rng: random.Random) -> Dict[str, Any]:
    def _one():
        # 3 parallel edges e1,e2,e3
        F = rint(rng, 35, 90)
        u = rint(rng, 10, 30)
        dcap = rint(rng, 5, 25)
        dcost = rint(rng, 1, 6)

        # base capacities
        caps = [rint(rng, max(u, 12), 55) for _ in range(3)]
        # base costs
        costs = [rint(rng, 1, 12) for _ in range(3)]

        # avoid trivial dominance: sort by cost but keep mixed capacities
        # enforce not all costs equal
        if len(set(costs)) == 1:
            raise ValueError("equal costs too often")

        best_cost, alloc = min_cost_parallel_flow(F, caps, costs, u, dcap, dcost)

        # ensure threshold is crossed in optimal solution for at least one edge
        if not any(fi > u for fi in alloc):
            raise ValueError("threshold never crossed in optimum")

        # ensure not all flow on one edge
        if sum(1 for fi in alloc if fi > 0) < 2:
            raise ValueError("single-edge solution")

        edges_str = "; ".join(
            [f"e{i+1}(cap={caps[i]}, cost={costs[i]})" for i in range(3)]
        )

        return {
            "EDGES": edges_str,
            "F": F,
            "u": u,
            "DeltaCap": dcap,
            "DeltaCost": dcost,
        }

    return tries(_one)


# -----------------------------
# Family 29: assignment with fairness penalty + hard-task group constraint
# -----------------------------
import itertools

def _validate_f29_binding_and_unique(
    n: int,
    C: list[list[int]],
    G1: set[int],
    HARD_SET: set[int],
    m: int,
    lam: int,
) -> None:
    """
    Raises ValueError if:
      1) the hard-task constraint is NOT binding (i.e., some unconstrained optimum already satisfies it), or
      2) the constrained optimum is NOT unique (multiple optimal assignments tie).
    """
    tasks = list(range(1, n + 1))

    def eval_perm(perm: tuple[int, ...]) -> tuple[int, int]:
        # perm[i] is task assigned to worker (i+1)
        total = 0
        sum_g1 = 0
        sum_g2 = 0
        hard_in_g1 = 0
        for wi, task in enumerate(perm, start=1):
            c = C[wi - 1][task - 1]
            total += c
            if wi in G1:
                sum_g1 += c
                if task in HARD_SET:
                    hard_in_g1 += 1
            else:
                sum_g2 += c
        obj = total + lam * abs(sum_g1 - sum_g2)
        return obj, hard_in_g1

    # Unconstrained optimum; track whether ANY unconstrained optimum satisfies the hard constraint
    best_u = None
    any_best_u_feasible = False

    # Constrained optimum; track uniqueness
    best_c = None
    count_best_c = 0

    for perm in itertools.permutations(tasks):
        obj, hard_in_g1 = eval_perm(perm)

        # unconstrained tracking
        if best_u is None or obj < best_u:
            best_u = obj
            any_best_u_feasible = (hard_in_g1 >= m)
        elif obj == best_u:
            if hard_in_g1 >= m:
                any_best_u_feasible = True

        # constrained tracking
        if hard_in_g1 >= m:
            if best_c is None or obj < best_c:
                best_c = obj
                count_best_c = 1
            elif obj == best_c:
                count_best_c += 1

    if best_c is None:
        raise ValueError("hard constraint infeasible (should not happen with current sampling)")

    # Check 1: binding (constraint must rule out all unconstrained optima)
    if any_best_u_feasible:
        raise ValueError("hard constraint not binding (unconstrained optimum already satisfies it)")

    # Check 2: uniqueness of constrained optimum (since prompt asks for 'the assignment')
    if count_best_c != 1:
        raise ValueError(f"constrained optimum not unique (ties={count_best_c})")



def sample_f29(rng: random.Random) -> Dict[str, Any]:
    def _one():
        n = rint(rng, 4, 7)
        workers = list(range(1, n + 1))
        tasks = list(range(1, n + 1))

        rng.shuffle(workers)
        g1_size = rint(rng, 1, n - 1)
        G1 = set(workers[:g1_size])
        G2 = set(workers[g1_size:])

        hard_sz = rint(rng, 1, max(1, n // 2))
        hard_tasks = tasks[:]
        rng.shuffle(hard_tasks)
        HARD_SET = set(hard_tasks[:hard_sz])

        m = rint(rng, 1, min(len(G1), len(HARD_SET)))
        lam = rint(rng, 1, 15)

        # cost matrix varied
        C = [[rint(rng, 1, 25) for _ in range(n)] for __ in range(n)]
        if len({C[i][j] for i in range(n) for j in range(n)}) < 6:
            raise ValueError("matrix too uniform")
        _validate_f29_binding_and_unique(n=n, C=C, G1=G1, HARD_SET=HARD_SET, m=m, lam=lam)

        # render matrix as compact text
        matrix_str = "[" + "; ".join([",".join(map(str, row)) for row in C]) + "]"

        return {
            "n": n,
            "COST_MATRIX": matrix_str,
            "lam": lam,
            "G1": set_str(list(G1)),
            "G2": set_str(list(G2)),
            "HARD_SET": set_str(list(HARD_SET)),
            "m": m,
        }

    return tries(_one)


# -----------------------------
# Family 30: count schedules with forbidden patterns (DP feasibility check)
# -----------------------------


def count_schedules(D: int, S: int, A: int, B: int, u: int) -> int:
    # DP over positions (day, slot), tracking:
    # - remaining X (A_rem), last_is_Y, x_count_in_day
    # Constraints:
    # - no consecutive Y across entire schedule (includes day boundary via last_is_Y)
    # - each day must have >= u X talks
    #
    # We'll compute counts, but cap growth (we only need nonzero).
    CAP = 10**18

    from functools import lru_cache

    @lru_cache(None)
    def dp(day: int, slot: int, A_rem: int, lastY: int, x_in_day: int) -> int:
        if day == D:
            # all days done
            return 1 if A_rem == 0 else 0

        if slot == S:
            # end of day: must meet daily minimum X
            if x_in_day < u:
                return 0
            # next day start
            return dp(day + 1, 0, A_rem, lastY, 0)

        pos_left = (D - day) * S - slot
        if A_rem < 0 or A_rem > pos_left:
            return 0

        total = 0
        # choose X
        if A_rem > 0:
            total += dp(day, slot + 1, A_rem - 1, 0, x_in_day + 1)
        # choose Y
        B_used = (day * S + slot) - ((A - A_rem))  # used positions minus used X
        B_rem = B - B_used
        if B_rem > 0 and lastY == 0:
            # placing Y, daily X count unchanged
            total += dp(day, slot + 1, A_rem, 1, x_in_day)

        return min(total, CAP)

    return dp(0, 0, A, 0, 0)


def count_schedules_no_adj(D: int, S: int, A: int, B: int, u: int) -> int:
    """
    Counts schedules with:
      - exactly A X talks and B Y talks (A+B = D*S),
      - each day has >= u X talks,
      - NO restriction on consecutive Y (adjacency constraint removed).
    This is used as an ablation check to ensure the adjacency constraint is actually binding.
    """
    CAP = 10**18
    from functools import lru_cache

    @lru_cache(None)
    def dp(day: int, slot: int, A_rem: int, x_in_day: int) -> int:
        if day == D:
            return 1 if A_rem == 0 else 0

        if slot == S:
            if x_in_day < u:
                return 0
            return dp(day + 1, 0, A_rem, 0)

        pos_used = day * S + slot
        pos_left = D * S - pos_used
        if A_rem < 0 or A_rem > pos_left:
            return 0

        # infer how many Y have been used so far
        A_used = A - A_rem
        Y_used = pos_used - A_used
        B_rem = B - Y_used
        if B_rem < 0:
            return 0

        total = 0
        # place X
        if A_rem > 0:
            total += dp(day, slot + 1, A_rem - 1, x_in_day + 1)
        # place Y (no adjacency restriction)
        if B_rem > 0:
            total += dp(day, slot + 1, A_rem, x_in_day)

        return min(total, CAP)

    return dp(0, 0, A, 0)


def sample_f30(rng: random.Random) -> Dict[str, Any]:
    def _one():
        D = rint(rng, 2, 4)
        S = rint(rng, 4, 8)
        total = D * S

        # FIX 1: make daily-minimum X constraint non-redundant.
        # For a length-S day with no consecutive Y, minimum X is floor(S/2).
        # So choose u > floor(S/2) to make it bind.
        u_lo = S // 2 + 1
        u_hi = S - 1
        if u_lo > u_hi:
            raise ValueError("no room for binding u")
        u = rint(rng, u_lo, u_hi)

        # global max Y under no-consecutive-Y across the whole schedule
        maxY_global = (total + 1) // 2
        # max Y allowed by daily minimum X: A >= D*u => B <= total - D*u
        maxY_daily = total - D * u
        B_max = min(maxY_global, maxY_daily)

        # FIX 2: avoid trivial B where adjacency constraint can't matter
        if B_max < 2:
            raise ValueError("B would be too small to make adjacency meaningful")

        # pick B nontrivially
        B = rint(rng, 2, B_max)
        A = total - B
        if A < D * u:
            raise ValueError("violates daily minimum X")

        cnt = count_schedules(D, S, A, B, u)
        if cnt == 0:
            raise ValueError("no valid schedules")

        # FIX 3: ablation checks to ensure constraints are actually binding

        # 3a) daily-min constraint binding: compare to u=0 with same adjacency rule
        cnt_u0 = count_schedules(D, S, A, B, 0)
        if cnt_u0 == cnt:
            raise ValueError("daily-min X constraint not binding")

        # 3b) adjacency constraint binding: compare to relaxed adjacency (no adjacency restriction)
        cnt_no_adj = count_schedules_no_adj(D, S, A, B, u)
        if cnt_no_adj == cnt:
            raise ValueError("adjacency constraint not binding")

        return {"D": D, "S": S, "A": A, "B": B, "u": u}

    return tries(_one)


# -----------------------------
# Families 23–30
# -----------------------------

FAMILIES: List[ProblemFamily] = [
    # ProblemFamily(
    #     "f23",
    #     "Project schedule with precedence, overtime, and penalty threshold",
    #     "A project has tasks A, B, C, D with durations dA={dA}, dB={dB}, dC={dC}, dD={dD}. Task B and C start after A; D starts after both B and C. You can pay overtime to reduce any single task by up to {r}% at cost {c} per hour reduced. If total completion time exceeds {T}, you pay penalty {P} per hour late, but if you finish at or before {T2} you receive bonus {B}. Find the minimum expected cost assuming overtime decisions are made after you learn that task C’s duration is actually dC+{Delta} (with probability {p}) otherwise dC.",
    #     sample_f23,
    # ),
    # ProblemFamily(
    #     "f24",
    #     "Bin packing with tiered shipping and must-split constraints",
    #     "You must ship {N} items with weights {weights}. Boxes have max capacity {Cap}. Shipping cost is {c1} per box for the first {m} boxes and {c2} thereafter, plus surcharge {S} if any box exceeds {theta} weight. Additionally, items in set {FRAGILE_SET} cannot share a box with items in {HEAVY_SET}. Find the minimum total shipping cost.",
    #     sample_f24,
    # ),
    # ProblemFamily(
    #     "f25",
    #     "Shortest path with time-dependent tolls and conditional detour",
    #     "A driver travels from {Start} to {End} through a directed graph with edges: {EDGES}. If the cumulative toll paid so far exceeds {TollCap}, all subsequent tolls are discounted by {d}%. However, if the driver enters node {X} before time {tau}, edge {e_star} becomes unavailable. Find the minimum possible total cost defined as {alpha}*(total time)+{beta}*(total toll).",
    #     sample_f25,
    # ),
    # ProblemFamily(
    #     "f26",
    #     "Inventory replenishment with lead time and stockout penalties",
    #     "Daily demand is {d1} units for days 1–{H1}, then {d2} units for days {H1_plus_1}–{H}. You start with {I0} units, can place an order any day which arrives after {L} days with fixed ordering cost {K} plus {c} per unit ordered. Holding cost is {h} per unit per day; stockout cost is {p} per unit unmet demand (lost sales). Additionally, if inventory ever falls below {S}, you must place an emergency order of {E} units (arrives in {Le} days at premium cost {cp}). Find the minimum total cost ordering policy under these rules.",
    #     sample_f26,
    # ),
    # ProblemFamily(
    #     "f27",
    #     "Counting with inclusion–exclusion plus at least/exactly mix",
    #     "In a class of {N} students, {a} take Math, {b} take Physics, {c} take CS. Exactly {ab} take Math and Physics, {ac} take Math and CS, {bc} take Physics and CS, and {abc} take all three. Additionally, {z} take none. Determine how many take exactly one subject and verify consistency of the data.",
    #     sample_f27,
    # ),
    # ProblemFamily(
    #     "f28",
    #     "Flow with capacity upgrades triggered by threshold usage",
    #     "A network has a single source and sink with parallel edges described by: {EDGES}. You must send {F} units. If any edge carries more than {u} units, its capacity increases by {DeltaCap} but its cost increases by {DeltaCost} for all units on that edge. Find the minimum total cost flow.",
    #     sample_f28,
    # ),
    # ProblemFamily(
    #     "f29",
    #     "Assignment with fairness constraint and penalty for imbalance",
    #     "{n} workers must be assigned to {n} tasks with base costs given by COST_MATRIX={COST_MATRIX} (row i is worker i, column j is task j). Total cost is the sum of assigned costs plus penalty {lam} times the absolute difference between total assigned cost for Group 1 workers {G1} and Group 2 workers {G2}. Additionally, at least {m} tasks in set {HARD_SET} must be assigned to Group 1. Find the assignment minimizing total cost.",
    #     sample_f29,
    # ),
    ProblemFamily(
        "f30",
        "Multi-stage counting of schedules with forbidden patterns",
        "A conference has {D} days and {S} sessions per day. You must schedule {A} talks of type X and {B} talks of type Y. Constraints: each day must include at least {u} X talks, no two Y talks can be consecutive within a day, and the first session of each day cannot be Y if the last session of the previous day was Y. Count the number of valid schedules.",
        sample_f30,
    ),
]


# -----------------------------
# Generation + outputs (same structure as first probstatgen)
# -----------------------------


def generate_instances(
    rng: random.Random, family: ProblemFamily, n: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    while len(out) < n:
        params = family.sampler(rng)
        key = tuple(sorted(params.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        text = render(family.template, params)
        out.append(
            {
                "family_id": family.family_id,
                "title": family.title,
                "params": params,
                "text": text,
            }
        )
    return out


def main(out_dir: str = "out_dmopt", n_per_family: int = 10, seed: int = 12345) -> None:
    rng = random.Random(seed)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    manifest = [
        {"family_id": f.family_id, "title": f.title, "template": f.template}
        for f in FAMILIES
    ]
    (outp / "families_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
