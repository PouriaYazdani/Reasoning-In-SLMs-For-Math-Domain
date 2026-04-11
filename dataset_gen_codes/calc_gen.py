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
    return round(rng.uniform(lo, hi), nd)


def tries(sampler, max_tries: int = 80_000) -> Dict[str, Any]:
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
# 31) Tank inflow/outflow with threshold-triggered outflow switch + capacity spill
# -----------------------------


def sample_f31(rng: random.Random) -> Dict[str, Any]:
    def _one():
        V0 = rint(rng, 20, 180)
        # Choose rates so: below H net positive; above H (during inflow) net still positive but smaller;
        # after inflow stops net negative and volume will drop below H again.
        a = rint(rng, 6, 22)  # inflow L/min
        b = rint(rng, 1, a - 2)  # outflow L/min (base)
        b2 = rint(rng, b + 1, a - 1)  # increased outflow, still < a

        T = rint(rng, 15, 120)  # minutes inflow is on

        # Ensure H is crossed during inflow
        H = rint(rng, V0 + 10, V0 + 120)
        net1 = a - b
        if net1 <= 0:
            raise ValueError("Need net positive below H.")
        tH = (H - V0) / net1
        if not (0 < tH < T - 1e-9):
            raise ValueError("Must cross H during inflow.")

        # Volume at end of inflow (t=T)
        net2 = a - b2
        if net2 <= 0:
            raise ValueError("Need net positive above H during inflow.")
        VT = H + net2 * (T - tH)  # once crossed, it grows with net2

        # Choose target and capacity so:
        # - Vtarget reachable
        # - spill nontrivial: Cap < VT
        Vtarget = rint(rng, int(H + 2), int(min(VT - 1, H + 80)))
        if Vtarget <= H:
            raise ValueError("Vtarget must exceed H.")

        Cap = rint(rng, int(max(Vtarget, H + 5)), int(max(Vtarget + 1, VT - 1)))
        if not (Vtarget <= Cap < VT):
            raise ValueError(
                "Need Vtarget <= Cap < VT for nonzero spill and reachable target."
            )

        return {
            "V0": V0,
            "a": a,
            "b": b,
            "T": T,
            "H": H,
            "b2": b2,
            "Vtarget": Vtarget,
            "Cap": Cap,
        }

    return tries(_one)


# -----------------------------
# 32) Cost integral with tiered pricing + hour-local surcharge if hour energy exceeds H
# -----------------------------


def hour_energy(p0: float, p1: float, j: int) -> float:
    # Energy in hour j (0-index), integral from j to j+1 of p0+p1 t dt = p0 + p1*(j+0.5)
    return p0 + p1 * (j + 0.5)


def total_energy(p0: float, p1: float, T: int) -> float:
    # integral from 0..T of p0+p1 t dt
    return p0 * T + 0.5 * p1 * T * T


def sample_f32(rng: random.Random) -> Dict[str, Any]:
    def _one():
        T = rint(rng, 4, 10)
        p0 = rfloat(rng, 1.0, 10.0, 2)
        p1 = rfloat(rng, 0.3, 4.0, 2)  # increasing

        hrs = [hour_energy(p0, p1, j) for j in range(T)]
        hmin, hmax = min(hrs), max(hrs)
        if hmax - hmin < 1.0:
            raise ValueError("Too flat; want some hours to cross threshold.")

        # Pick H between early and late hour energies to ensure both sides occur
        H = rfloat(rng, hmin + 0.2 * (hmax - hmin), hmin + 0.8 * (hmax - hmin), 2)

        above = sum(1 for e in hrs if e > H)
        below = T - above
        if not (above >= 1 and below >= 1):
            raise ValueError("Need some hours above and some below the hour threshold.")

        E_total = total_energy(p0, p1, T)
        # Tier boundary E should be crossed but not be extreme
        E = rfloat(rng, 0.35 * E_total, 0.75 * E_total, 2)
        if not (0.0 < E < E_total):
            raise ValueError("Tier boundary must be within total usage.")

        c1 = rfloat(rng, 0.08, 0.22, 3)
        c2 = rfloat(rng, max(c1 + 0.01, 0.12), 0.40, 3)
        s = rfloat(rng, 0.02, 0.20, 3)

        return {"T": T, "p0": p0, "p1": p1, "c1": c1, "E": E, "c2": c2, "H": H, "s": s}

    return tries(_one)


# -----------------------------
# 33) Constrained optimization with piecewise penalty
# -----------------------------
# -----------------------------
# 33) Constrained optimization with piecewise penalty (REPLACEMENT)
#   Enforces:
#     - At least one feasibility constraint binds (cap or emissions).
#     - The maximizer is at the active feasibility bound.
#     - The quadratic penalty region is active at the maximizer (x > U).
#   (Rounding-precision caveats intentionally ignored.)
# -----------------------------


def sample_f33(rng: random.Random) -> Dict[str, Any]:
    def _argmax_with_penalty(
        ra: int,
        rb: float,
        ca: int,
        p: int,
        L: float,
        q: float,
        U: float,
        xmax: float,
    ) -> Tuple[float, float]:
        # base profit: (ra-ca)x - rb x^2
        # penalty: +p if x<L; +q(x-U)^2 if x>U; else 0
        def obj(x: float) -> float:
            base = (ra - ca) * x - rb * x * x
            if x < L:
                return base - p
            if x > U:
                return base - q * (x - U) * (x - U)
            return base

        # candidate points (exact for piecewise concave regions)
        cand = set()

        # boundaries
        cand.add(0.0)
        cand.add(max(0.0, min(xmax, L)))
        cand.add(max(0.0, min(xmax, U)))
        cand.add(xmax)

        # interior critical point in [L, U]: x_star (of base concave)
        x_star = (ra - ca) / (2.0 * rb)
        if L <= x_star <= U:
            cand.add(x_star)
        else:
            cand.add(max(L, min(U, x_star)))

        # interior critical point in (U, xmax]: derivative zero for base - q(x-U)^2
        # d/dx = (ra-ca) - 2rb x - 2q(x-U) = 0
        xu = ((ra - ca) + 2.0 * q * U) / (2.0 * (rb + q))
        if U < xu < xmax:
            cand.add(xu)

        best_x = None
        best_v = None
        for x in cand:
            x = float(max(0.0, min(xmax, x)))
            v = obj(x)
            if best_v is None or v > best_v + 1e-12:
                best_v = v
                best_x = x

        return float(best_x), float(best_v)

    def _one():
        # Base profit concave: R(x)=ra x - rb x^2, C(x)=ca x
        ra = rint(rng, 60, 220)
        rb = rfloat(rng, 0.8, 6.0, 2)
        ca = rint(rng, 5, ra - 10)

        x_star = (ra - ca) / (2.0 * rb)
        if not (8.0 < x_star < 80.0):
            raise ValueError("Need a moderate interior x*.")

        # Make feasibility bind: choose active xmax < x_star
        # Decide whether cap or emissions is the binding constraint
        bind_emissions = rng.random() < 0.5

        # Choose an effective max in a strict range below x_star
        xmax = rfloat(rng, max(4.0, 0.55 * x_star), 0.90 * x_star, 2)
        if not (xmax < x_star - 1.0):
            raise ValueError("Need xmax to bind (xmax < x*).")

        if bind_emissions:
            # emissions binds: x <= em_lim / e = xmax
            cap = rint(rng, int(math.ceil(xmax + 2.0)), int(math.ceil(xmax + 40.0)))
            e = rfloat(rng, 0.5, 3.0, 2)
            em_lim = rfloat(rng, e * xmax, e * xmax, 2)  # lock exact
            x_em_max = em_lim / e
            if abs(x_em_max - xmax) > 1e-9:
                raise ValueError("Emissions max mismatch.")
        else:
            # capacity binds: x <= cap = xmax (as an integer bound)
            cap = int(round(xmax))
            if cap < 4:
                raise ValueError("cap too small.")
            xmax = float(cap)  # make the bound exact
            e = rfloat(rng, 0.5, 3.0, 2)
            # set emissions looser than capacity
            slack = rfloat(rng, 5.0, 40.0, 2)
            em_lim = rfloat(rng, e * (xmax + slack), e * (xmax + slack), 2)
            x_em_max = em_lim / e
            if x_em_max <= xmax + 1e-6:
                raise ValueError("Need emissions not binding when capacity binds.")

        # Penalty thresholds; enforce U < xmax so quadratic penalty is active at maximizer
        gap = rfloat(rng, 0.6, min(8.0, 0.25 * xmax), 2)
        U = round(xmax - gap, 2)
        if not (1.0 < U < xmax - 0.2):
            raise ValueError("Need U strictly below xmax.")

        L = rfloat(rng, max(0.5, U - 14.0), U - 2.0, 2)
        if not (0.0 < L < U):
            raise ValueError("Need 0 < L < U.")

        p = rint(rng, 20, 400)

        # Choose q so the penalized objective is still increasing up to xmax
        # i.e., derivative at xmax is positive:
        # d = (ra-ca) - 2rb*xmax - 2q*(xmax-U) > 0
        numer = (ra - ca) - 2.0 * rb * xmax
        if numer <= 0.5:
            raise ValueError("Need base increasing at xmax (x* must exceed xmax).")
        denom = 2.0 * (xmax - U)
        q_max = 0.85 * (numer / denom)
        if q_max <= 0.5:
            raise ValueError("q_max too small; adjust U/xmax spacing.")
        q = rfloat(rng, 0.5, min(20.0, q_max), 2)

        # Ensure penalty magnitude is nontrivial at xmax
        if q * (xmax - U) * (xmax - U) < 3.0:
            raise ValueError("Quadratic penalty too small at xmax.")

        # Acceptance test: maximizer must be at xmax and in x>U region (penalty active)
        x_best, _ = _argmax_with_penalty(ra, rb, ca, p, L, q, U, xmax)
        if abs(x_best - xmax) > 1e-6:
            raise ValueError("Maximizer not at feasibility bound.")
        if not (x_best > U + 1e-9):
            raise ValueError("Penalty region not active at maximizer.")

        # Text expressions
        R_expr = f"{ra}x - {rb}x^2"
        C_expr = f"{ca}x"
        g1_expr = f"x - {cap}"
        g2_expr = f"{e}x - {em_lim}"

        return {
            "R_expr": R_expr,
            "C_expr": C_expr,
            "p": p,
            "L": L,
            "q": q,
            "U": U,
            "g1_expr": g1_expr,
            "g2_expr": g2_expr,
        }

    return tries(_one)



# -----------------------------
# 34) Exponential decay with periodic dosing + skip-next-dose rule on threshold exceed
# -----------------------------


def simulate_dosing_skip(
    n: int, D: float, k: float, Delta: float, H: float, Tend: float
) -> Tuple[float, int]:
    # scheduled times t_i = (i-1)*Delta for i=1..n
    t = 0.0
    C = 0.0
    skip_next = False
    skipped = 0

    for i in range(n):
        t_i = i * Delta
        # decay from previous event time to this dose time
        C *= math.exp(-k * (t_i - t))
        t = t_i

        if skip_next:
            skipped += 1
            skip_next = False
            continue

        # administer dose
        C += D
        if C > H:
            skip_next = True

    # decay from last scheduled time to Tend (assume Tend >= last time)
    if Tend < t:
        raise ValueError("Tend must be >= last scheduled dose time.")
    C *= math.exp(-k * (Tend - t))
    return C, skipped


def sample_f34(rng: random.Random) -> Dict[str, Any]:
    def _one():
        n = rint(rng, 5, 12)
        Delta = rfloat(rng, 2.0, 10.0, 2)  # hours between scheduled doses
        k = rfloat(rng, 0.04, 0.35, 3)  # 1/hour
        D = rfloat(rng, 5.0, 40.0, 2)  # dose units

        f = math.exp(-k * Delta)
        if not (0.15 < f < 0.90):
            raise ValueError("Need moderate decay between doses.")

        steady_peak = D / (1.0 - f)
        # Choose H so skips can occur but not always
        H = rfloat(rng, 0.60 * steady_peak, 0.95 * steady_peak, 2)

        Tend = rfloat(
            rng, (n - 1) * Delta + 0.5 * Delta, (n - 1) * Delta + 2.0 * Delta, 2
        )

        C_end, skipped = simulate_dosing_skip(n, D, k, Delta, H, Tend)

        # enforce "skipping can occur but not always": require at least one skip, but not skipping almost everything
        if not (1 <= skipped <= max(1, n // 2)):
            raise ValueError("Skip count not in desired range.")

        Total = round(n * D, 2)  # planned total dose (before skip rule)

        return {
            "k": k,
            "H": H,
            "n": n,
            "D": D,
            "Delta": Delta,
            "Total": Total,
            "Tend": Tend,
        }

    return tries(_one)


# -----------------------------
# 35) Ladder related rates with piecewise motion and conditional stop
# -----------------------------


def sample_f35(rng: random.Random) -> Dict[str, Any]:
    def _one():
        L = rfloat(rng, 4.0, 15.0, 2)  # meters
        v1 = rfloat(rng, 0.10, 0.80, 2)  # m/s
        v2 = rfloat(rng, 0.10, 1.20, 2)  # m/s
        T = rint(rng, 3, 20)  # seconds
        Sstop = rint(rng, 2, 15)  # seconds

        x1 = v1 * T
        if x1 >= L - 0.2:
            raise ValueError("Bottom too far; ladder would be nearly horizontal.")

        y1 = math.sqrt(max(0.0, L * L - x1 * x1))

        # target top height h (must be lower than y1 so it happens after time T)
        h = rfloat(rng, 0.20 * L, min(0.85 * L, y1 - 0.10), 2)
        if not (0.0 < h < y1):
            raise ValueError("Need target height below y(T).")

        x_target = math.sqrt(max(0.0, L * L - h * h))
        if x_target <= x1 + 0.15:
            raise ValueError("Would reach target during phase 1; want multi-stage.")

        # Choose whether stop triggers; embed it implicitly via hmin and y(T)
        trigger_stop = rng.random() < 0.5

        if trigger_stop:
            hmin = rfloat(
                rng, max(h + 0.05, y1 + 0.05), min(L - 0.01, y1 + 0.40), 2
            )  # y1 < hmin
            if not (y1 < hmin):
                raise ValueError("Stop must trigger.")
        else:
            hmin = rfloat(
                rng, max(h + 0.05, y1 - 0.40), max(h + 0.06, y1 - 0.05), 2
            )  # y1 >= hmin
            if not (y1 >= hmin):
                raise ValueError("Stop must not trigger.")

        # Ensure phase 2 time is reasonable
        t2 = (x_target - x1) / v2
        if not (0.5 <= t2 <= 200.0):
            raise ValueError("Phase 2 duration unreasonable.")

        return {"L": L, "v1": v1, "T": T, "v2": v2, "hmin": hmin, "S": Sstop, "h": h}

    return tries(_one)


# -----------------------------
# Families 31–35
# -----------------------------

FAMILIES: List[ProblemFamily] = [
    # ProblemFamily(
    #     "f31",
    #     "Piecewise inflow/outflow with threshold-triggered rate switch",
    #     "A tank initially contains {V0} liters. Inflow is {a} L/min for the first {T} minutes (then 0). Outflow is {b} L/min throughout, but if volume ever exceeds {H}, outflow increases to {b2} L/min until volume drops below {H} again. Compute the time when the volume first hits {Vtarget}, and the total amount spilled if the tank capacity is {Cap}.",
    #     sample_f31,
    # ),
    # ProblemFamily(
    #     "f32",
    #     "Cost integral with tiered pricing and a constraint on average rate",
    #     "A machine runs for {T} hours with power consumption P(t) = {p0} + {p1}t kW. Electricity price is {c1} per kWh up to {E} kWh total usage, then {c2} beyond, and if usage during any single hour exceeds {H}, an extra surcharge {s} per kWh applies to that hour only. Compute total cost and the average price per kWh.",
    #     sample_f32,
    # ),
    ProblemFamily(
        "f33",
        "Optimization with piecewise penalty and feasibility interval",
        "A company chooses production level x (continuous) with revenue R(x)={R_expr} and base cost C(x)={C_expr}. The cost includes a piecewise penalty: add {p} if x<{L}, add {q}(x-{U})^2 if x>{U}, otherwise 0. Additionally, x must satisfy g1(x)={g1_expr}<=0 and g2(x)={g2_expr}<=0. Find the maximizing x and the profit.",
        sample_f33,
    ),
    # ProblemFamily(
    #     "f34",
    #     "Differential equation with intervention at threshold and total-dose constraint",
    #     "A drug concentration follows dC/dt = -{k}C between doses. You administer {n} identical bolus doses of size {D} at scheduled times t_i=(i-1)*{Delta} hours for i=1..{n}. If concentration exceeds {H}, you skip the next scheduled dose once. The planned total dose is exactly {Total}. Given k, H, and this schedule pattern, compute the final concentration at time {Tend} and how many doses were skipped.",
    #     sample_f34,
    # ),
    # ProblemFamily(
    #     "f35",
    #     "Related rates with multi-stage motion and constraint change",
    #     "A ladder of length {L} slides with the bottom moving away from the wall at {v1} m/s for {T} seconds. After that, the bottom speed changes to {v2} unless the top height is below {hmin}, in which case the ladder is stopped for {S} seconds then resumes at {v2}. Find the total time until the top reaches height {h} and the total distance the bottom traveled.",
    #     sample_f35,
    # ),
]


# -----------------------------
# Generation + outputs
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


def main(out_dir: str = "out_calc", n_per_family: int = 10, seed: int = 12345) -> None:
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
