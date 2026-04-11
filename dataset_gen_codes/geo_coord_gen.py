# geo_coord_gen.py
# Python 3.10+
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

def tries(sampler, max_tries: int = 120_000) -> Dict[str, Any]:
    last = None
    for _ in range(max_tries):
        try:
            return sampler()
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to sample valid instance. Last error: {last}")

def sign_term(x: float, nd: int = 3) -> str:
    v = round(x, nd)
    return f"+{v}" if v >= 0 else f"{v}"

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Geometry helpers
# -----------------------------

def segment_area_above_xaxis(R: float, y0: float) -> float:
    # area of circle portion with y >= 0 for circle radius R centered at (x0, y0)
    d = abs(y0)
    full = math.pi * R * R
    if d >= R:
        return full if y0 > 0 else 0.0
    # segment area on the "far" side of the chord (shrinks to 0 when tangent)
    seg = R * R * math.acos(d / R) - d * math.sqrt(max(0.0, R * R - d * d))
    return full - seg if y0 >= 0 else seg

def shoelace_area(poly: List[Tuple[float, float]]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0

def perimeter(poly: List[Tuple[float, float]]) -> float:
    per = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        per += math.hypot(x2 - x1, y2 - y1)
    return per

def segments_intersect(a, b, c, d) -> bool:
    # proper intersection for segments ab and cd (including touching)
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    eps = 1e-12
    if abs(o1) < eps and on_segment(a, c, b): return True
    if abs(o2) < eps and on_segment(a, d, b): return True
    if abs(o3) < eps and on_segment(c, a, d): return True
    if abs(o4) < eps and on_segment(c, b, d): return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def is_simple_quadrilateral(A, B, C, D) -> bool:
    # A-B-C-D polygon: check AB intersects CD or BC intersects DA
    if segments_intersect(A, B, C, D):
        return False
    if segments_intersect(B, C, D, A):
        return False
    return True


# -----------------------------
# 36) Coordinate geometry + branch + circle segment area
# -----------------------------

def sample_f36(rng: random.Random) -> Dict[str, Any]:
    def _one():
        # choose whether condition |AB| < T triggers
        small_AB_branch = (rng.random() < 0.5)

        xP = rfloat(rng, -8.0, 8.0, 2)
        # keep |yP| not too tiny to avoid overly symmetric cases
        yP = rfloat(rng, -6.0, 6.0, 2)
        if abs(yP) < 0.6:
            yP = 0.6 if yP >= 0 else -0.6

        # pick a radius r so the circle intersects x-axis: |yP| < r
        if small_AB_branch:
            r = rfloat(rng, abs(yP) + 0.15, abs(yP) + 1.8, 3)
        else:
            r = rfloat(rng, abs(yP) + 2.0, abs(yP) + 7.0, 3)

        # chord length on x-axis
        AB = 2.0 * math.sqrt(max(0.0, r * r - yP * yP))
        if AB <= 0.05:
            raise ValueError("degenerate intersection with x-axis")

        if small_AB_branch:
            T = rfloat(rng, AB + 0.2, AB + 5.0, 2)
        else:
            T = rfloat(rng, max(0.2, AB - 5.0), AB - 0.2, 2)
            if T <= 0:
                raise ValueError("T must be positive")

        # choose slopes and derive intercepts so intersection is at P
        m1 = rfloat(rng, -4.0, 4.0, 2)
        m2 = rfloat(rng, -4.0, 4.0, 2)
        if abs(m1 - m2) < 0.4:
            raise ValueError("slopes too close")

        b1 = yP - m1 * xP
        b2 = yP - m2 * xP

        # branch parameters
        r2 = rfloat(rng, max(abs(yP) + 0.10, 0.3), max(abs(yP) + 4.0, 1.0), 3)
        if r2 <= 0:
            raise ValueError("r2 positive")

        # translation should keep segment computation meaningful; keep new center within intersection range
        dx = rfloat(rng, -5.0, 5.0, 2)
        # choose dy so |yP+dy| < r for intersection with x-axis (often)
        dy = rfloat(rng, -r + 0.25 - yP, r - 0.25 - yP, 2)

        # ensure both radii lead to defined segment area (always), but keep reasonable
        if abs(yP) >= r:
            raise ValueError("must intersect x-axis initially")

        return {
            "m1": m1, "b1_term": sign_term(b1, 2),
            "m2": m2, "b2_term": sign_term(b2, 2),
            "r": r, "T": T, "r2": r2, "dx": dx, "dy": dy
        }
    return tries(_one)


# -----------------------------
# 37) Triangle with ratio + perimeter + area + altitude threshold scaling
# -----------------------------

def heron_area(a: float, b: float, c: float) -> float:
    s = 0.5 * (a + b + c)
    v = max(0.0, s * (s - a) * (s - b) * (s - c))
    return math.sqrt(v)

def sample_f37(rng: random.Random) -> Dict[str, Any]:
    def _one():
        trigger_scale = (rng.random() < 0.5)

        # choose coprime-ish small ratio
        p = rint(rng, 2, 8)
        q = rint(rng, 2, 8)
        if math.gcd(p, q) != 1:
            raise ValueError("prefer reduced ratio")

        k = rint(rng, 4, 22)
        a = k * p
        b = k * q

        # choose c to form a non-degenerate triangle
        c_lo = abs(a - b) + 1
        c_hi = a + b - 1
        if c_lo >= c_hi:
            raise ValueError("no feasible c")

        c = rint(rng, c_lo, min(c_hi, c_lo + 60))

        A = heron_area(a, b, c)
        if A <= 1.0:
            raise ValueError("area too small")

        P = a + b + c

        # altitude to side a
        ha = 2.0 * A / a

        # choose h0 to enforce desired branch robustly
        if trigger_scale:
            h0 = rfloat(rng, ha + 0.15 * ha + 0.05, ha + 0.60 * ha + 0.20, 3)  # h0 > ha
            if not (ha < h0 - 1e-6):
                raise ValueError("scale must trigger")
        else:
            h0 = rfloat(rng, max(0.05, ha - 0.60 * ha), ha - 0.15 * ha - 0.05, 3)  # h0 < ha
            if not (ha > h0 + 1e-6):
                raise ValueError("scale must not trigger")

        s_factor = rfloat(rng, 1.10, 2.30, 3)

        # uniqueness check (w.r.t. rounded area) among integer k consistent with same p:q and same perimeter P
        A_round = round(A, 3)
        matches = 0
        for kk in range(1, 60):
            aa = kk * p
            bb = kk * q
            cc = P - aa - bb
            if cc <= 0:
                continue
            if not (abs(aa - bb) < cc < aa + bb):
                continue
            AA = heron_area(aa, bb, cc)
            if round(AA, 3) == A_round:
                matches += 1
                if matches > 1:
                    raise ValueError("non-unique triangle under rounding")
        if matches != 1:
            raise ValueError("no matching triangle under rounding")

        return {"p": p, "q": q, "P": P, "A": A_round, "h0": h0, "s": s_factor}
    return tries(_one)


def sample_f38(rng: random.Random) -> Dict[str, Any]:
    def _one():
        trigger_rotate = (rng.random() < 0.5)

        R = rfloat(rng, 5.0, 30.0, 3)

        # choose eps so it is meaningfully inside the circle
        eps = rfloat(rng, 0.5, 0.45 * R, 3)

        # choose d1 to force branch, but do NOT return it (solver must infer from R and c1)
        if trigger_rotate:
            # keep d1 away from 0 to reduce sensitivity to c1 rounding
            hi = min(0.75 * eps, R - 0.05)
            lo = 0.20
            if lo >= hi:
                raise ValueError("no feasible d1 range for rotate branch")
            d1 = rfloat(rng, lo, hi, 3)
        else:
            lo = min(0.95 * R, eps + 0.20)
            hi = 0.85 * R
            if lo >= hi:
                raise ValueError("no feasible d1 range for shorten branch")
            d1 = rfloat(rng, lo, hi, 3)

        if not (0.0 < d1 < R):
            raise ValueError("d1 must be in (0,R)")

        # chord 1 length consistent with R and d1
        c1 = 2.0 * math.sqrt(max(0.0, R * R - d1 * d1))
        c1 = round(c1, 3)
        if c1 <= 0.2:
            raise ValueError("c1 too small")

        # Robust branch check using inferred distance from displayed (rounded) c1:
        # inferred d1_hat = sqrt(R^2 - (c1/2)^2)
        d1_hat = math.sqrt(max(0.0, R * R - (c1 / 2.0) * (c1 / 2.0)))
        margin = 0.05  # keeps inequality unambiguous despite rounding

        if trigger_rotate and not (d1_hat < eps - margin):
            raise ValueError("rotation branch must trigger under inferred d1")
        if (not trigger_rotate) and not (d1_hat > eps + margin):
            raise ValueError("shorten branch must trigger under inferred d1")

        # chord 2: pick a distance d2 and set c2 consistently
        d2 = rfloat(rng, 0.05, 0.90 * R, 3)
        c2 = 2.0 * math.sqrt(max(0.0, R * R - d2 * d2))
        c2 = round(c2, 3)
        if c2 <= 0.2:
            raise ValueError("c2 too small")

        # shortening amount for chord 1 (used only when not rotating)
        Delta = rfloat(rng, 0.20, min(0.45 * c1, 8.0), 3)
        if c1 - Delta <= 0.2:
            raise ValueError("shortened chord too small")

        # IMPORTANT: do not return d1
        return {"R": R, "c1": c1, "c2": c2, "eps": eps, "d2": d2, "Delta": Delta}

    return tries(_one)

# -----------------------------
# 39) Cylinder fill with discrete slosh loss above height H at each minute mark
# -----------------------------

def simulate_cyl_slosh(
    r_m: float, T1: int, a1: float, T2: int, a2: float, b: float,
    H_m: float, s_pct: float
) -> Tuple[float, float, bool]:
    area = math.pi * r_m * r_m  # m^2
    V = 0.0  # liters
    lost = 0.0
    crossed = False
    total_minutes = T1 + T2
    V_H = area * H_m * 1000.0  # liters (since 1 m^3 = 1000 L)

    for minute in range(1, total_minutes + 1):
        a = a1 if minute <= T1 else a2
        V += (a - b)  # liters over 1 minute
        if V < 0:
            return 0.0, 0.0, False  # invalid for our sampling goals

        # slosh at each minute mark if above H
        if V > V_H + 1e-9:
            crossed = True
            excess = V - V_H
            loss = (s_pct / 100.0) * excess
            V -= loss
            lost += loss

    h_final = V / (1000.0 * area)
    return h_final, lost, crossed

def sample_f39(rng: random.Random) -> Dict[str, Any]:
    def _one():
        r = rfloat(rng, 0.25, 1.50, 3)  # meters
        T1 = rint(rng, 10, 40)
        T2 = rint(rng, 10, 40)

        a1 = rfloat(rng, 10.0, 80.0, 2)
        a2 = rfloat(rng, 5.0, 80.0, 2)
        b = rfloat(rng, 0.0, min(a1 - 1.0, 65.0), 2)  # ensure phase1 net positive

        H = rfloat(rng, 0.20, 2.50, 3)
        s = rfloat(rng, 1.0, 18.0, 2)

        h_final, lost, crossed = simulate_cyl_slosh(r, T1, a1, T2, a2, b, H, s)
        if not crossed:
            raise ValueError("sloshing never triggers")
        if lost <= 0.1:
            raise ValueError("lost volume too small")
        if h_final < 0.0:
            raise ValueError("negative height")
        # avoid extreme outcomes
        if h_final > 6.0:
            raise ValueError("height too large")

        return {"r": r, "T1": T1, "a1": a1, "T2": T2, "a2": a2, "b": b, "H": H, "s": s}
    return tries(_one)


# -----------------------------
# 40) Quadrilateral area branch -> shift C or reflect D across y=k, then perimeter+area
# -----------------------------

def sample_convex_quad(rng: random.Random) -> List[Tuple[float, float]]:
    # generate 4 points around a center, sort by angle
    cx = rfloat(rng, -5.0, 5.0, 2)
    cy = rfloat(rng, -5.0, 5.0, 2)
    pts = []
    angles = sorted([rng.uniform(0, 2 * math.pi) for _ in range(4)])
    for ang in angles:
        rad = rng.uniform(2.0, 7.0)
        x = cx + rad * math.cos(ang)
        y = cy + rad * math.sin(ang)
        pts.append((round(x, 2), round(y, 2)))
    return pts

def sample_f40(rng: random.Random) -> Dict[str, Any]:
    def _one():
        shift_branch = (rng.random() < 0.5)  # area > A0 -> shift C

        # build a simple quad in order
        for _ in range(200):
            poly = sample_convex_quad(rng)
            A, B, C, D = poly
            if not is_simple_quadrilateral(A, B, C, D):
                continue
            area0 = shoelace_area(poly)
            if area0 < 4.0:
                continue

            if shift_branch:
                A0 = round(area0 - rng.uniform(0.6, min(3.0, 0.35 * area0)), 3)
                u = rfloat(rng, -2.0, 2.0, 2)
                v = rfloat(rng, -2.0, 2.0, 2)
                C2 = (round(C[0] + u, 2), round(C[1] + v, 2))
                poly2 = [A, B, C2, D]
                if is_simple_quadrilateral(*poly2) and shoelace_area(poly2) > 1e-6:
                    # choose any k (unused in this branch but present)
                    k = rfloat(rng, -5.0, 5.0, 2)
                    return {
                        "x1": A[0], "y1": A[1],
                        "x2": B[0], "y2": B[1],
                        "x3": C[0], "y3": C[1],
                        "x4": D[0], "y4": D[1],
                        "A0": A0, "u": u, "v": v, "k": k
                    }
            else:
                A0 = round(area0 + rng.uniform(0.6, min(3.0, 0.35 * area0)), 3)
                # pick k near y4 so reflection doesn't wildly distort
                k = rfloat(rng, D[1] - 1.5, D[1] + 1.5, 2)
                D2 = (D[0], round(2 * k - D[1], 2))
                poly2 = [A, B, C, D2]
                if is_simple_quadrilateral(*poly2) and shoelace_area(poly2) > 1e-6:
                    # choose u,v (unused in this branch but present)
                    u = rfloat(rng, -2.0, 2.0, 2)
                    v = rfloat(rng, -2.0, 2.0, 2)
                    return {
                        "x1": A[0], "y1": A[1],
                        "x2": B[0], "y2": B[1],
                        "x3": C[0], "y3": C[1],
                        "x4": D[0], "y4": D[1],
                        "A0": A0, "u": u, "v": v, "k": k
                    }

        raise ValueError("could not sample a valid simple quad")
    return tries(_one)


# -----------------------------
# Families 36–40
# -----------------------------

FAMILIES: List[ProblemFamily] = [
    # ProblemFamily(
    #     "f36",
    #     "Coordinate geometry with intersection constraint + conditional replacement",
    #     "Two lines ℓ1: y={m1}x{b1_term} and ℓ2: y={m2}x{b2_term} intersect at P. A circle centered at P with radius {r} intersects the x-axis at points A and B. If |AB|<{T}, the radius is replaced by {r2}; otherwise the circle is translated by vector ({dx},{dy}). Compute the final area of the circle segment above the x-axis.",
    #     sample_f36,
    # ),
    # ProblemFamily(
    #     "f37",
    #     "Triangle with mixed constraints (side ratio + area + altitude threshold)",
    #     "A triangle has sides a,b,c with a:b = {p}:{q} and perimeter {P}. Its area is {A}. If the altitude to side a is less than {h0}, the triangle is scaled uniformly by factor {s}. Find the final side lengths and the final inradius.",
    #     sample_f37,
    # ),
    # ProblemFamily(
    #     "f38",
    #     "Circle chord geometry with two-stage constraint update",
    #     "In a circle of radius {R}, two chords have lengths {c1} and {c2} and intersect at point P inside the circle "
    #     "(assume P is the midpoint of chord 1). If P lies within {eps} of the center, chord 2 is rotated to make its "
    #     "distance from center {d2}; otherwise chord 1 is shortened by {Delta}. Compute the final intersection power "
    #     "PA·PB for chord 1.",
    #     sample_f38,
    # ),
    # ProblemFamily(
    #     "f39",
    #     "3D volume with piecewise filling and sloshing loss",
    #     "A cylindrical tank of radius {r} is initially empty and is filled with water. For the first {T1} minutes, inflow is {a1} L/min; for the next {T2} minutes it is {a2} L/min. Outflow is {b} L/min throughout. At the end of each minute (after applying that minute’s inflow and outflow), if the water height exceeds {H}, a sloshing loss of {s}% of the excess volume above height {H} occurs instantly. Find the final water height and the total lost volume.",
    #     sample_f39,
    # ),
    ProblemFamily(
        "f40",
        "Coordinate polygon area with constraint-dependent vertex shift",
        "A quadrilateral has vertices A({x1},{y1}), B({x2},{y2}), C({x3},{y3}), D({x4},{y4}). If its area (shoelace) exceeds {A0}, shift vertex C by vector ({u},{v}); otherwise reflect vertex D across line y={k}. Compute the final perimeter and area.",
        sample_f40,
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


def main(out_dir: str = "out_geometry", n_per_family: int = 10, seed: int = 12345) -> None:
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
