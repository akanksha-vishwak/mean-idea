"""Build the communication package directly from raw experiment evidence."""

from __future__ import annotations

import hashlib
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPERIMENTS = ROOT / "experiments"
ASSETS = HERE / "assets"
PPTX_PATH = HERE / "presentation.pptx"
POSTER_PATH = HERE / "poster.html"
REPORT_PATH = HERE / "technical-report.html"
ANSWERS_PATH = HERE / "answers-to-sergiy.md"
BUILD_DATE = "September 18, 2026"
PRIOR_PILOT_RUNS_CANDIDATES = [
    HERE / "evidence" / "prior-pilot",
]

NAVY = "#102A43"
BLUE = "#2563EB"
CYAN = "#06B6D4"
PURPLE = "#7C3AED"
LAVENDER = "#EDE9FE"
SKY = "#E0F2FE"
LIGHT = "#F8FAFC"
MID = "#64748B"
DARK = "#0F172A"
GREEN = "#15803D"
RED = "#B91C1C"
AMBER = "#B45309"
WHITE = "#FFFFFF"
PALETTE = [BLUE, PURPLE, CYAN, "#4F46E5", "#0EA5E9"]


class EvidenceError(RuntimeError):
    pass


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceError(f"Mandatory evidence is missing: {display_path(path)}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"Invalid JSON in {display_path(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected a JSON object in {display_path(path)}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_dir(number: int) -> Path:
    matches = sorted(EXPERIMENTS.glob(f"{number:03d}-*"))
    require(len(matches) == 1, f"Expected exactly one Experiment {number:03d} directory; found {len(matches)}")
    return matches[0]


def load_evidence() -> dict[int, dict[str, Any]]:
    evidence: dict[int, dict[str, Any]] = {}
    for number in range(1, 14):
        directory = experiment_dir(number)
        results_path = directory / "results" / "results.json"
        spec_path = directory / "spec.json"
        require(results_path.exists(), f"Mandatory evidence is missing: {results_path.relative_to(ROOT)}")
        require(spec_path.exists(), f"Mandatory experiment specification is missing: {spec_path.relative_to(ROOT)}")
        evidence[number] = {
            "number": number,
            "dir": directory,
            "results_path": results_path,
            "spec_path": spec_path,
            "results": load_json(results_path),
            "spec": load_json(spec_path),
        }
        require(
            evidence[number]["results"].get("experiment_id") == directory.name,
            f"Experiment ID mismatch in {results_path.relative_to(ROOT)}",
        )
        require(
            str(evidence[number]["results"].get("status", "")).startswith("completed"),
            f"Experiment {number:03d} is not complete in {results_path.relative_to(ROOT)}",
        )
    return evidence


def load_prior_scaffold_evidence() -> dict[str, Any]:
    for runs_root in PRIOR_PILOT_RUNS_CANDIDATES:
        partial_dir = runs_root / "partial-scaffolds-20260916"
        handoff_dir = runs_root / "scaffold-handoff-20260916"
        required = [
            partial_dir / "results.json",
            partial_dir / "assessment.json",
            handoff_dir / "results.json",
            handoff_dir / "assessment.json",
        ]
        if all(path.exists() for path in required):
            return {
                "partial_results": load_json(required[0]),
                "partial_assessment": load_json(required[1]),
                "handoff_results": load_json(required[2]),
                "handoff_assessment": load_json(required[3]),
                "source_paths": [display_path(path) for path in required],
            }
    searched = ", ".join(display_path(path) for path in PRIOR_PILOT_RUNS_CANDIDATES)
    raise EvidenceError(
        "Mandatory earlier-pilot scaffold evidence is missing. "
        f"Searched these runs directories: {searched}"
    )


def find_raw_text(exp: dict[str, Any], filename: str, preferred_subdir: str | None = None) -> Path:
    result_dir = exp["dir"] / "results"
    if preferred_subdir:
        candidate = result_dir / preferred_subdir / filename
        if candidate.exists():
            return candidate
    matches = list(result_dir.rglob(filename))
    require(matches, f"Referenced raw text is missing: {exp['dir'].name}/results/**/{filename}")
    if len(matches) > 1:
        raise EvidenceError(
            f"Ambiguous raw text reference {filename} in {exp['dir'].name}; "
            f"found {len(matches)} files and no matching preferred subdirectory"
        )
    return matches[0]


def read_raw_text(exp: dict[str, Any], filename: str, preferred_subdir: str | None = None) -> tuple[str, str]:
    path = find_raw_text(exp, filename, preferred_subdir)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    require(bool(text), f"Raw output is empty: {path.relative_to(ROOT)}")
    return text, path.relative_to(ROOT).as_posix()


def excerpt(text: str, limit: int = 340) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def indicators(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("denoised_lexical_indicators") or run.get("lexical_indicators") or {}


def calculate_claims(e: dict[int, dict[str, Any]], prior: dict[str, Any]) -> dict[str, Any]:
    r4 = e[4]["results"]
    runs4 = r4.get("runs", [])
    require(r4.get("endpoint_controls_passed") is True, "Experiment 004 endpoint controls did not pass")
    intermediate4 = [r for r in runs4 if 0 < float(r["weight"]) < 1]
    require(len(intermediate4) == 27, f"Experiment 004 expected 27 intermediate runs, found {len(intermediate4)}")
    by_weight4: dict[float, Counter[str]] = {}
    for run in intermediate4:
        ind = indicators(run)
        category = "hybrid" if ind.get("retains_both") else "A" if ind.get("retains_A") else "B" if ind.get("retains_B") else "unrelated"
        by_weight4.setdefault(float(run["weight"]), Counter())[category] += 1
    hybrids4 = sum(bool(indicators(run).get("candidate_hybrid")) for run in intermediate4)
    require(hybrids4 == 0, f"Experiment 004 raw fields report {hybrids4} intermediate hybrids, expected 0")

    r5 = e[5]["results"]
    geometry = r5["parent_geometry"]
    require(geometry["global_cosine"] > 0, "Experiment 005 global cosine is not positive")
    require(geometry["position_cosine"]["negative_position_count"] == 0, "Experiment 005 has negative position cosines")

    r6 = e[6]["results"]
    midpoint6 = next(run for run in r6["runs"] if math.isclose(float(run["weight"]), 0.5))
    availability6 = midpoint6["summary"]["availability_by_cutoff"]
    require(availability6["1"]["both_present_count"] == 0, "Experiment 006 midpoint unexpectedly has both endpoints at top-1")

    r7 = e[7]["results"]
    midpoint7 = [r for r in r7["runs"] if math.isclose(float(r["weight"]), 0.5)]
    require(len(midpoint7) == 15, f"Experiment 007 expected 15 midpoint runs, found {len(midpoint7)}")
    hybrids7 = sum(bool(indicators(r).get("candidate_hybrid")) for r in midpoint7)
    endpoint7 = [r for r in r7["runs"] if float(r["weight"]) in (0.0, 1.0)]
    endpoint_pass7 = all(
        indicators(r).get("retains_A") if float(r["weight"]) == 0 else indicators(r).get("retains_B")
        for r in endpoint7
    )
    require(endpoint_pass7, "Experiment 007 endpoint controls do not all pass")

    r8 = e[8]["results"]
    dual8 = [
        r for r in r8["runs"]
        if r["method"] == "dual_prompt_logit_mixing" and 0 < float(r["weight"]) < 1
    ]
    hard8 = [r for r in r8["runs"] if r["method"] == "aligned_hard_projection"]
    direct8 = [r for r in r8["runs"] if r["method"] == "direct_diffusion_prompt"]
    require((len(dual8), len(hard8), len(direct8)) == (9, 3, 3), "Experiment 008 run counts differ from the controlled design")

    r9 = e[9]["results"]
    valid9 = [pair for pair in r9["pair_results"] if pair.get("endpoint_controls_passed") is True]
    blocked9 = [pair for pair in r9["pair_results"] if pair.get("endpoint_controls_passed") is False]
    require(len(valid9) == 2 and len(blocked9) == 1, "Experiment 009 endpoint-gate counts differ from the design")
    numerical9 = [
        run for pair in valid9 for run in pair["runs"]
        if run["method"] != "direct_diffusion_prompt" and math.isclose(float(run.get("weight", 0.5)), 0.5)
    ]
    direct9 = [run for pair in valid9 for run in pair["runs"] if run["method"] == "direct_diffusion_prompt"]
    require((len(numerical9), len(direct9)) == (12, 6), "Experiment 009 controlled run counts differ from the design")

    r10 = e[10]["results"]
    reproduction10 = r10["original_b_reproduction_runs"]
    strong10 = r10["strengthened_endpoint_runs"]
    repro_pass10 = sum(bool(r["canonical_reproduction_match"]) for r in reproduction10)
    strong_a10 = [r for r in strong10 if float(r["weight"]) == 0.0]
    strong_b10 = [r for r in strong10 if float(r["weight"]) == 1.0]
    strong_a_pass10 = sum(bool(indicators(r).get("retains_A")) for r in strong_a10)
    strong_b_pass10 = sum(bool(indicators(r).get("retains_B")) for r in strong_b10)
    require((repro_pass10, strong_a_pass10, strong_b_pass10) == (3, 1, 0), "Experiment 010 control counts differ from raw evidence")

    r11 = e[11]["results"]
    passing11 = [pair for pair in r11["pair_results"] if pair.get("pair_endpoint_gate_passed") is True]
    blocked11 = [pair for pair in r11["pair_results"] if pair.get("pair_endpoint_gate_passed") is False]
    require(
        (len(passing11), len(blocked11)) == (3, 1),
        "Experiment 011 endpoint screen does not contain three passing and one blocked pair",
    )
    require(
        [pair["pair_id"] for pair in passing11]
        == ["matrix-blocked-blas", "graph-csr-deque-bfs", "image-batched-vectorization"],
        "Experiment 011 passing-pair IDs differ from the frozen selection",
    )
    require(blocked11[0]["pair_id"] == "regex-mmap-scan", "Experiment 011 blocked pair is not regex-mmap-scan")

    r12 = e[12]["results"]
    require(r12["summary"]["completed_run_count"] == 27, "Experiment 012 did not complete all 27 frozen runs")
    require(
        r12["reference"]["passing_pair_ids"] == [pair["pair_id"] for pair in passing11],
        "Experiment 012 pair selection does not match Experiment 011's passing endpoint gates",
    )
    runs12 = [run for pair in r12["pair_results"] for run in pair["runs"]]
    dual12 = [run for run in runs12 if run["method"] == "dual_prompt_logit_mixing"]
    hard12 = [run for run in runs12 if run["method"] == "aligned_hard_projection"]
    direct12 = [run for run in runs12 if run["method"] == "direct_diffusion_prompt"]
    require((len(dual12), len(hard12), len(direct12)) == (9, 9, 9), "Experiment 012 method counts differ from the frozen design")
    direct12_by_pair = {
        pair["pair_id"]: (
            sum(bool(indicators(run).get("candidate_hybrid")) for run in pair["runs"] if run["method"] == "direct_diffusion_prompt"),
            sum(run["method"] == "direct_diffusion_prompt" for run in pair["runs"]),
        )
        for pair in r12["pair_results"]
    }
    require(
        direct12_by_pair
        == {
            "matrix-blocked-blas": (3, 3),
            "graph-csr-deque-bfs": (2, 3),
            "image-batched-vectorization": (3, 3),
        },
        "Experiment 012 per-pair direct-prompt strict counts differ from raw evidence",
    )

    r13 = e[13]["results"]
    require(
        r13["settings"]["parent_derived_canvas_state"] is False,
        "Experiment 013 unexpectedly used a parent-derived canvas state",
    )
    runs13 = [
        run
        for pair in r13["pair_results"]
        for run in pair["runs"]
        if run["method"] == "full_context_blank_canvas_prompt"
    ]
    require(
        len(runs13) == 9
        and all(run.get("status") == "completed" for run in runs13),
        "Experiment 013 did not complete all nine frozen runs",
    )
    full13_by_pair = {
        pair["pair_id"]: (
            sum(
                bool(indicators(run).get("candidate_hybrid"))
                for run in pair["runs"]
            ),
            len(pair["runs"]),
        )
        for pair in r13["pair_results"]
    }
    require(
        full13_by_pair
        == {
            "matrix-blocked-blas": (3, 3),
            "graph-csr-deque-bfs": (3, 3),
            "image-batched-vectorization": (2, 3),
        },
        "Experiment 013 per-pair strict counts differ from raw evidence",
    )
    full13_hybrids = sum(
        bool(indicators(run).get("candidate_hybrid")) for run in runs13
    )
    require(
        (
            full13_hybrids,
            r13["summary"]["concise_baseline_hybrid_count"],
            r13["summary"]["concise_baseline_run_count"],
        )
        == (8, 8, 9),
        "Experiment 013 does not report the reviewed 8/9 versus 8/9 comparison",
    )
    failed13 = [
        (pair["pair_id"], run)
        for pair in r13["pair_results"]
        for run in pair["runs"]
        if not indicators(run).get("candidate_hybrid")
    ]
    require(
        len(failed13) == 1
        and failed13[0][0] == "image-batched-vectorization"
        and failed13[0][1]["seed"] == 44
        and indicators(failed13[0][1]).get("retains_A") is True
        and indicators(failed13[0][1]).get("retains_B") is False,
        "Experiment 013 strict failure is not the reviewed image seed 44 output",
    )

    numerical8 = dual8 + hard8
    numerical_controlled = numerical8 + numerical9 + dual12 + hard12
    direct_controlled = direct8 + direct9 + direct12
    numerical_hybrids = sum(bool(indicators(r).get("candidate_hybrid")) for r in numerical_controlled)
    direct_hybrids = sum(bool(indicators(r).get("candidate_hybrid")) for r in direct_controlled)
    require(
        (len(numerical_controlled), numerical_hybrids, len(direct_controlled), direct_hybrids) == (42, 0, 18, 17),
        "Experiments 008+009+012 aggregate does not equal 0/42 numerical versus 17/18 direct prompting",
    )

    partial_runs = prior["partial_results"].get("runs", [])
    partial_assessment = prior["partial_assessment"]
    require(
        len(partial_runs) == 15 and all(run.get("status") == "completed" for run in partial_runs),
        "Earlier partial-scaffold pilot does not contain 15 completed runs",
    )
    require(
        partial_assessment.get("useful_partial_scaffolds_found") == 0
        and partial_assessment.get("parent_concepts_found") == 0,
        "Earlier partial-scaffold assessment does not report zero useful scaffolds and zero parent concepts",
    )
    handoff_runs = prior["handoff_results"].get("runs", [])
    handoff_assessment = prior["handoff_assessment"]
    require(
        len(handoff_runs) == 8 and all(run.get("status") == "completed" for run in handoff_runs),
        "Earlier scaffold-handoff pilot does not contain eight completed runs",
    )
    require(
        handoff_assessment.get("scaffold_only", {}).get("parent_a_recovered") == "0/4"
        and handoff_assessment.get("scaffold_only", {}).get("parent_b_recovered") == "0/4"
        and handoff_assessment.get("parents_and_scaffold", {}).get("both_parents_recovered") == "4/4"
        and "3/3" in handoff_assessment.get("parents_and_scaffold", {}).get("comparison", ""),
        "Earlier scaffold-handoff assessment differs from the reviewed 0/4, 4/4, and parents-only 3/3 comparison",
    )

    return {
        "exp4_intermediate_count": len(intermediate4),
        "exp4_hybrids": hybrids4,
        "exp4_by_weight": by_weight4,
        "geometry": geometry,
        "exp6_availability": availability6,
        "exp6_distinct_positions": midpoint6["summary"]["endpoint_distinct_position_count"],
        "exp7_midpoints": len(midpoint7),
        "exp7_hybrids": hybrids7,
        "exp7_endpoint_pass": endpoint_pass7,
        "exp8_dual": (sum(bool(indicators(r).get("candidate_hybrid")) for r in dual8), len(dual8)),
        "exp8_hard": (sum(bool(indicators(r).get("candidate_hybrid")) for r in hard8), len(hard8)),
        "exp8_direct": (sum(bool(indicators(r).get("candidate_hybrid")) for r in direct8), len(direct8)),
        "exp9_valid_pairs": len(valid9),
        "exp9_blocked_pairs": len(blocked9),
        "exp9_numerical": (sum(bool(indicators(r).get("candidate_hybrid")) for r in numerical9), len(numerical9)),
        "exp9_direct": (sum(bool(indicators(r).get("candidate_hybrid")) for r in direct9), len(direct9)),
        "exp10_reproduction": (repro_pass10, len(reproduction10)),
        "exp10_strong_a": (strong_a_pass10, len(strong_a10)),
        "exp10_strong_b": (strong_b_pass10, len(strong_b10)),
        "exp11_pair_count": len(r11["pair_results"]),
        "exp11_passing": len(passing11),
        "exp11_passing_ids": [pair["pair_id"] for pair in passing11],
        "exp11_blocked_ids": [pair["pair_id"] for pair in blocked11],
        "exp12_run_count": len(runs12),
        "exp12_dual": (sum(bool(indicators(r).get("candidate_hybrid")) for r in dual12), len(dual12)),
        "exp12_hard": (sum(bool(indicators(r).get("candidate_hybrid")) for r in hard12), len(hard12)),
        "exp12_direct": (sum(bool(indicators(r).get("candidate_hybrid")) for r in direct12), len(direct12)),
        "exp12_direct_by_pair": direct12_by_pair,
        "exp13_full": (full13_hybrids, len(runs13)),
        "exp13_concise": (
            r13["summary"]["concise_baseline_hybrid_count"],
            r13["summary"]["concise_baseline_run_count"],
        ),
        "exp13_full_by_pair": full13_by_pair,
        "controlled_numerical": (numerical_hybrids, len(numerical_controlled)),
        "controlled_direct": (direct_hybrids, len(direct_controlled)),
        "partial_scaffold_runs": len(partial_runs),
        "partial_scaffold_useful": partial_assessment["useful_partial_scaffolds_found"],
        "handoff_scaffold_only": "0/4",
        "handoff_parents_and_scaffold": "4/4",
        "handoff_parents_only": "3/3",
        "prior_pilot_sources": prior["source_paths"],
    }


def collect_samples(e: dict[int, dict[str, Any]]) -> dict[str, tuple[str, str]]:
    samples: dict[str, tuple[str, str]] = {}
    for key, number, filename, subdir in [
        ("e1_mix", 1, "a-b-mixture.txt", None),
        ("e2_b", 2, "b-control.txt", None),
        ("e3_explicit", 3, "explicit-source-only-steps-8.txt", None),
        ("e4_mid", 4, "weight-0p5-seed-42.txt", None),
        ("e5_mid", 5, "weight-0p5-projection.txt", None),
        ("e6_mid", 6, "weight-0p5-top1.txt", None),
        ("e7_mid", 7, "temperature-1-weight-0p5-seed-42-denoised.txt", None),
        ("e8_mid", 8, "weight-0p5-seed-42-denoised.txt", None),
        ("e8_direct", 8, "direct-prompt-seed-42.txt", None),
        ("e9_mid", 9, "dual-weight-0p5-seed-42-denoised.txt", "pair-fast-path-fallback"),
        ("e9_direct", 9, "direct-prompt-seed-42.txt", "pair-fast-path-fallback"),
        ("e10_a", 10, "dual-weight-0p0-seed-44-denoised.txt", "strengthened-aligned"),
        ("e11_a", 11, "dual-weight-0p0-seed-42-denoised.txt", "matrix-blocked-blas"),
        ("e11_b", 11, "dual-weight-1p0-seed-42-denoised.txt", "matrix-blocked-blas"),
        ("e12_mid", 12, "dual-weight-0p5-seed-42-denoised.txt", "matrix-blocked-blas"),
        ("e12_direct", 12, "direct-prompt-seed-42.txt", "matrix-blocked-blas"),
        ("e12_graph_damaged", 12, "direct-prompt-seed-43.txt", "graph-csr-deque-bfs"),
        ("e13_graph", 13, "full-context-seed-42.txt", "graph-csr-deque-bfs"),
        ("e13_image_strict_failure", 13, "full-context-seed-44.txt", "image-batched-vectorization"),
    ]:
        samples[key] = read_raw_text(e[number], filename, subdir)
    return samples


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": DARK,
            "xtick.color": MID,
            "ytick.color": MID,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )


def save_plot(fig: plt.Figure, filename: str, alt: str, alt_map: dict[str, str]) -> Path:
    path = ASSETS / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    alt_map[filename] = alt
    return path


def build_plots(c: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    configure_plots()
    ASSETS.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}
    alt: dict[str, str] = {}

    weights = sorted(c["exp4_by_weight"])
    categories = ["A", "unrelated", "B", "hybrid"]
    colors = [BLUE, "#94A3B8", PURPLE, GREEN]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    bottoms = [0] * len(weights)
    for category, color in zip(categories, colors):
        values = [c["exp4_by_weight"][w][category] for w in weights]
        ax.bar(weights, values, width=0.075, bottom=bottoms, label=category, color=color)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_title("Experiment 004: discrete basins, no hybrid region")
    ax.set_xlabel("Interpolation weight (0 = Parent A, 1 = Parent B)")
    ax.set_ylabel("Outputs across 3 seeds")
    ax.set_xticks(weights)
    ax.set_ylim(0, 3.25)
    ax.legend(ncol=4, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    plots["exp004"] = save_plot(
        fig,
        "exp004-weight-regions.png",
        "Stacked bars across weights 0.1 through 0.9 show outputs classified as Parent A, unrelated, or Parent B; none are hybrids.",
        alt,
    )

    g = c["geometry"]
    names = ["Global cosine", "Minimum position cosine", "Mean position cosine", "Midpoint norm retained"]
    values = [
        g["global_cosine"],
        g["position_cosine"]["minimum"],
        g["position_cosine"]["mean"],
        g["midpoint_norm_retention_vs_mean_parent_norm"],
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    bars = ax.bar(names, values, color=[BLUE, CYAN, "#0EA5E9", PURPLE])
    ax.set_ylim(0, 1.05)
    ax.set_title("Experiment 005: the midpoint does not collapse geometrically")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=10)
    plots["exp005"] = save_plot(
        fig,
        "exp005-geometry.png",
        (
            "Four bars show positive global and position cosine similarities "
            f"and {100 * g['midpoint_norm_retention_vs_mean_parent_norm']:.1f} "
            "percent midpoint norm retention."
        ),
        alt,
    )

    cutoffs = ["1", "5", "20"]
    a_vals = [c["exp6_availability"][k]["A_present_fraction"] for k in cutoffs]
    b_vals = [c["exp6_availability"][k]["B_present_fraction"] for k in cutoffs]
    both_vals = [c["exp6_availability"][k]["both_present_fraction"] for k in cutoffs]
    x = list(range(len(cutoffs)))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.25
    ax.bar([v - width for v in x], a_vals, width, label="A available", color=BLUE)
    ax.bar(x, b_vals, width, label="B available", color=PURPLE)
    ax.bar([v + width for v in x], both_vals, width, label="Both available", color=CYAN)
    ax.set_xticks(x, [f"Top-{k}" for k in cutoffs])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1.08)
    ax.set_title("Experiment 006 midpoint: information exists below top-1")
    ax.set_ylabel(f"Fraction of {c['exp6_distinct_positions']} endpoint-distinct positions")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    plots["exp006"] = save_plot(
        fig,
        "exp006-topk-availability.png",
        "Grouped bars show zero positions with both endpoint tokens at top-1, 52.5 percent at top-5, and 95 percent at top-20.",
        alt,
    )

    labels = [
        "Exp007\nstochastic",
        "Exp008\ndual logits",
        "Exp008\nhard",
        "Exp009\nnumerical",
        "Exp012\ndual logits",
        "Exp012\nhard",
        "Direct prompts\n008+009+012",
    ]
    pairs = [
        (c["exp7_hybrids"], c["exp7_midpoints"]),
        c["exp8_dual"],
        c["exp8_hard"],
        c["exp9_numerical"],
        c["exp12_dual"],
        c["exp12_hard"],
        c["controlled_direct"],
    ]
    rates = [success / total for success, total in pairs]
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    bars = ax.bar(labels, rates, color=["#94A3B8", BLUE, "#4F46E5", PURPLE, CYAN, "#0EA5E9", GREEN])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Hybrid success: numerical manipulation versus direct instruction")
    ax.set_ylabel("Candidate hybrid rate")
    ax.grid(axis="y", alpha=0.2)
    for bar, (success, total) in zip(bars, pairs):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0.02) + 0.035, f"{success}/{total}", ha="center", fontweight="bold")
    plots["methods"] = save_plot(
        fig,
        "method-comparison.png",
        "Bar chart shows zero hybrids for all tested numerical methods and 17 of 18 strict hybrids for direct prompts across Experiments 008, 009, and 012.",
        alt,
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    labels2 = ["Numerical methods", "Direct diffusion prompt"]
    successes = [c["controlled_numerical"][0], c["controlled_direct"][0]]
    totals = [c["controlled_numerical"][1], c["controlled_direct"][1]]
    failures = [t - s for s, t in zip(successes, totals)]
    ax.bar(labels2, failures, color="#CBD5E1", label="Not hybrid")
    ax.bar(labels2, successes, bottom=failures, color=[BLUE, GREEN], label="Hybrid")
    ax.set_title("Endpoint-valid pairs across four workloads")
    ax.set_ylabel("Outputs")
    ax.set_ylim(0, max(totals) + 3)
    for i, (s, t) in enumerate(zip(successes, totals)):
        ax.text(i, t + 0.6, f"{s}/{t} hybrids", ha="center", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    plots["aggregate"] = save_plot(
        fig,
        "controlled-aggregate.png",
        "Stacked bars compare zero of 42 hybrids from numerical representation methods with 17 of 18 strict hybrids from direct diffusion prompting.",
        alt,
    )
    return plots, alt


def h(text: Any) -> str:
    return html.escape(str(text))


def provenance_rows(e: dict[int, dict[str, Any]]) -> str:
    rows = []
    for number in range(1, 14):
        item = e[number]
        results = item["results"]
        status = results.get("status")
        revision = results.get("repository_revision", "not recorded in manifest")
        model = results.get("model_id") or results.get("model", {}).get("id", "not recorded")
        rows.append(
            f"<tr><td>{number:03d}</td><td>{h(item['dir'].name)}</td><td>{h(status)}</td>"
            f"<td><code>{h(revision)}</code></td><td>{h(model)}</td>"
            f"<td><code>{h(item['results_path'].relative_to(ROOT).as_posix())}</code></td></tr>"
        )
    return "\n".join(rows)


CSS = f"""
:root{{--navy:{NAVY};--blue:{BLUE};--purple:{PURPLE};--cyan:{CYAN};--light:{LIGHT};--mid:{MID};--dark:{DARK};}}
*{{box-sizing:border-box}} body{{margin:0;font-family:"Segoe UI",Arial,sans-serif;color:var(--dark);background:#fff;line-height:1.48}}
header{{background:linear-gradient(120deg,var(--navy),var(--blue) 58%,var(--purple));color:white;padding:2.2rem 4vw}}
h1{{font-size:2.35rem;margin:.1rem 0}} h2{{color:var(--navy);font-size:1.55rem;border-bottom:3px solid #dbeafe;padding-bottom:.35rem}}
h3{{color:var(--purple)}} main{{max-width:1180px;margin:auto;padding:1.4rem 3vw 3rem}}
.lede{{font-size:1.18rem;max-width:900px}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}
.card{{border:1px solid #dbe3ef;border-radius:14px;padding:1rem 1.15rem;background:#fff;box-shadow:0 3px 14px #0f172a0b}}
.callout{{border-left:7px solid var(--purple);background:#f5f3ff;padding:1rem 1.2rem;border-radius:8px;margin:1rem 0}}
.result{{font-size:1.45rem;font-weight:700;color:var(--navy)}} .small{{font-size:.84rem;color:var(--mid)}}
img.plot{{width:100%;height:auto;border:1px solid #e2e8f0;border-radius:10px;background:white}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}} th,td{{border:1px solid #dbe3ef;padding:.55rem;vertical-align:top}} th{{background:#eaf2ff;text-align:left}}
code{{font-family:Consolas,monospace;font-size:.86em;overflow-wrap:anywhere}} blockquote{{margin:.7rem 0;padding:.8rem 1rem;border-left:5px solid var(--cyan);background:#f0fdfa}}
.flow{{display:flex;gap:.6rem;align-items:stretch;flex-wrap:wrap;margin:1rem 0}} .flow div{{flex:1;min-width:145px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:.8rem;text-align:center;font-weight:600}}
.arrow{{align-self:center;font-size:1.5rem;color:var(--purple)}} footer{{padding:1rem 4vw;background:#eef2ff;color:var(--mid)}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
"""


def raw_quote(sample: tuple[str, str], limit: int = 360) -> str:
    text, path = sample
    return f"<blockquote>{h(excerpt(text, limit))}<div class='small'><code>{h(path)}</code></div></blockquote>"


def experiment_cards(e: dict[int, dict[str, Any]], c: dict[str, Any]) -> str:
    observations = {
        1: "Both parent techniques were recognizable in controls, but all three generated programs were syntactically invalid; the mixed code state did not combine them.",
        2: "Prompt-state controls produced coherent ideas, but Parent B lost its NumPy/bincount/repeat specificity, so this was not yet a valid mixing test.",
        3: "Explicit source wording restored NumPy, bincount, and repeat from 1 through 8 denoising steps. Prompt specificity was therefore a critical control.",
        4: f"Both endpoints passed. Across {c['exp4_intermediate_count']} intermediate runs, {c['exp4_hybrids']} hybrids appeared; outputs occupied A-like, unrelated, and B-like regions.",
        5: f"Global cosine was {c['geometry']['global_cosine']:.3f}; no position had negative cosine; midpoint norm retention was {c['geometry']['midpoint_norm_retention_vs_mean_parent_norm']:.1%}. Simple cancellation was not the explanation.",
        6: f"At the midpoint, both endpoint token alternatives were present in top-20 at {c['exp6_availability']['20']['both_present_fraction']:.1%} of endpoint-distinct positions, but never together at top-1. Hard projection assembled a patchwork.",
        7: f"Endpoint controls passed at all temperatures. Stochastic top-20 projection produced {c['exp7_hybrids']}/{c['exp7_midpoints']} midpoint hybrids.",
        8: f"Endpoints passed. Dual-logit intermediates: {c['exp8_dual'][0]}/{c['exp8_dual'][1]}; hard midpoint: {c['exp8_hard'][0]}/{c['exp8_hard'][1]}; direct diffusion prompts: {c['exp8_direct'][0]}/{c['exp8_direct'][1]}.",
        9: f"{c['exp9_valid_pairs']} sorting pairs passed endpoint gates and one histogram pair was blocked. Numerical midpoints: {c['exp9_numerical'][0]}/{c['exp9_numerical'][1]}; direct prompts: {c['exp9_direct'][0]}/{c['exp9_direct'][1]}.",
        10: f"Deterministic reproduction passed {c['exp10_reproduction'][0]}/{c['exp10_reproduction'][1]}. Strengthened A passed {c['exp10_strong_a'][0]}/{c['exp10_strong_a'][1]} and B {c['exp10_strong_b'][0]}/{c['exp10_strong_b'][1]}, so no midpoint test was admissible.",
        11: f"{c['exp11_passing']}/{c['exp11_pair_count']} nonsorting pairs passed both endpoint gates: matrix multiplication, graph BFS, and image normalization. The regex pair was blocked.",
        12: f"All {c['exp12_run_count']} frozen runs completed. Dual-logit midpoint: {c['exp12_dual'][0]}/{c['exp12_dual'][1]}; hard midpoint: {c['exp12_hard'][0]}/{c['exp12_hard'][1]}; direct prompting: {c['exp12_direct'][0]}/{c['exp12_direct'][1]} strict hybrids.",
        13: f"Full-context blank-canvas prompting produced {c['exp13_full'][0]}/{c['exp13_full'][1]} strict hybrids, tying the frozen concise-prompt baseline at {c['exp13_concise'][0]}/{c['exp13_concise'][1]}. Graph improved while image exactness declined.",
    }
    cards = []
    for number in range(1, 14):
        spec = e[number]["spec"]
        question = spec.get("question", "See experiment specification.")
        cards.append(
            f"<section class='card'><h3>Experiment {number:03d}: {h(e[number]['dir'].name.split('-',1)[1].replace('-', ' ').title())}</h3>"
            f"<p><strong>Question.</strong> {h(question)}</p><p><strong>Finding.</strong> {h(observations[number])}</p>"
            f"<p class='small'>Evidence: <code>{h(e[number]['results_path'].relative_to(ROOT).as_posix())}</code></p></section>"
        )
    return "\n".join(cards)


def build_poster(e: dict[int, dict[str, Any]], c: dict[str, Any], plots: dict[str, Path]) -> None:
    screen_summary = (
        f"Experiment 011: {c['exp11_passing']}/{c['exp11_pair_count']} endpoint-valid pairs "
        f"(matrix, graph, image); regex blocked. Experiment 012 tested numerical and concise-prompt "
        f"composition; Experiment 013's full-context blank-canvas prompt tied the concise baseline "
        f"at {c['exp13_full'][0]}/{c['exp13_full'][1]}."
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Can AI Average Two Ideas?</title>
<style>{CSS}
@page{{size:landscape;margin:0.35in}} body{{background:#eef2ff}} .poster{{width:16in;min-height:8.8in;margin:auto;background:white;padding:.35in .45in}}
.poster h1{{font-size:30pt;color:white}} .poster h2{{font-size:18pt;margin:.25rem 0}} .poster p,.poster li{{font-size:10.5pt}}
.poster header{{margin:-.35in -.45in .25in;padding:.25in .4in}} .poster-grid{{display:grid;grid-template-columns:1.08fr 1.2fr 1.08fr;gap:.16in}}
.poster .card{{padding:.12in;border-radius:8px;box-shadow:none}} .poster img{{max-height:2.15in;object-fit:contain}} .poster .result{{font-size:20pt}}
@media print{{body{{background:white}} .poster{{margin:0;width:auto;min-height:auto}}}}
</style></head><body><article class="poster"><header><h1>Can AI Average Two Optimization Ideas?</h1>
<p class="lede">A controlled pilot study of DiffusionGemma representations for Project Darwin idea synthesis</p></header>
<div class="poster-grid"><div>
<section class="card"><h2>Research question</h2><p>Can a diffusion language model combine two code-optimization ideas by averaging or mixing its internal prompt-conditioned representations?</p>
<p><strong>Success:</strong> a coherent idea that measurably retains both named parents.</p></section>
<section class="card"><h2>Why it matters</h2><p>Project Darwin explores program improvements. A useful numerical “idea space” could support controlled search for new optimizations—if nearby numerical points decode to related, valid ideas.</p></section>
<section class="card"><h2>Method</h2><div class="flow"><div>Parent A prompt</div><span class="arrow">+</span><div>Parent B prompt</div><span class="arrow">→</span><div>Mix states / logits</div><span class="arrow">→</span><div>Project + denoise</div></div>
<p>Controls first: endpoints had to reproduce each parent. Then we tested weights, geometry, top-token alternatives, stochastic projection, per-step logit mixing, multiple pairs, and prompt specificity.</p></section>
<section class="card"><h2>Evidence standard</h2><p>Every number on this poster is calculated at build time from Experiments 001–013 raw JSON fields. Raw output text is read directly.</p></section>
</div><div>
<section class="card"><h2>Strongest result</h2><p class="result">{c['controlled_numerical'][0]}/{c['controlled_numerical'][1]} numerical hybrids vs {c['controlled_direct'][0]}/{c['controlled_direct'][1]} direct-prompt hybrids</p>
<img class="plot" src="{plots['aggregate'].relative_to(HERE).as_posix()}" alt="Across endpoint-valid sorting, matrix, graph, and image pairs, zero of 42 numerical outputs were hybrids while 17 of 18 direct prompts were strict hybrids."></section>
<section class="card"><h2>The interpolation coordinate was not smooth</h2><img class="plot" src="{plots['exp004'].relative_to(HERE).as_posix()}" alt="Experiment 004 weight sweep showing Parent A, unrelated, and Parent B regions with no hybrids.">
<p>Experiment 004: endpoints passed, but {c['exp4_hybrids']}/{c['exp4_intermediate_count']} intermediate runs were hybrids.</p></section>
<section class="card"><h2>Why? A projection bottleneck</h2><img class="plot" src="{plots['exp006'].relative_to(HERE).as_posix()}" alt="At the midpoint, both endpoint alternatives are absent together at top-1 but widely available within top-20.">
<p>The continuous midpoint retained alternatives below rank 1, while independent hard token selection produced a semantically broken patchwork.</p></section>
</div><div>
<section class="card"><h2>What we ruled out</h2><img class="plot" src="{plots['exp005'].relative_to(HERE).as_posix()}" alt="Positive cosine and strong norm retention show that vector cancellation does not explain the failure.">
<p>No negative position cosines and {c['geometry']['midpoint_norm_retention_vs_mean_parent_norm']:.1%} norm retention make simple vector cancellation an unlikely explanation.</p></section>
<section class="card"><h2>What succeeded</h2><ul><li>DiffusionGemma decoded validated endpoints.</li><li>Explicit prompts restored missing implementation details.</li><li>Direct diffusion instructions composed both parents {c['controlled_direct'][0]}/{c['controlled_direct'][1]} strict times across endpoint-valid pairs.</li><li>Full-context blank-canvas prompting also achieved {c['exp13_full'][0]}/{c['exp13_full'][1]}, tying the concise baseline.</li><li>Raw manifests captured model, settings, seeds, runtime, and hashes.</li></ul></section>
<section class="card"><h2>Limitations</h2><ul><li>Exploratory/pilot—not confirmatory.</li><li>One model revision and public synthetic tasks.</li><li>Lexical rules are necessary but not a full human evaluation.</li><li>Internal-layer injection, KV-cache mixing, learned sequence-aware mixing, and preregistered confirmation are our later future-work ideas, not Sergiy’s explicit asks.</li></ul></section>
<section class="card"><h2>Conclusion</h2><p>The tested final-state interpolation, hard projection, and linear output-logit mixing did not create a useful semantic bridge across sorting, matrix multiplication, sparse graph BFS, and image normalization. Bayesian optimization over this coordinate is not justified. Do not integrate this numerical mixer into Darwin yet; keep the successful direct-prompt baseline.</p>
<p><strong>Our later future work:</strong> freeze a confirmatory protocol and test technically distinct sequence-aware mechanisms.</p><p class="small">{h(screen_summary)}</p></section>
</div></div><p class="small">Pilot evidence generated September 17–18, 2026. Build date: {BUILD_DATE}. Full provenance: technical-report.html.</p></article></body></html>"""
    POSTER_PATH.write_text(document, encoding="utf-8")


def build_report(
    e: dict[int, dict[str, Any]],
    c: dict[str, Any],
    plots: dict[str, Path],
    samples: dict[str, tuple[str, str]],
) -> None:
    model_revisions = sorted(
        {
            item["results"].get("model", {}).get("revision")
            for item in e.values()
            if item["results"] and item["results"].get("model", {}).get("revision")
        }
    )
    sample_sections = f"""
    <h2>Representative raw-output excerpts</h2>
    <p>These excerpts are loaded from raw text files during each build; they are not cached evidence.</p>
    <div class="grid">
      <section class="card"><h3>Exp001 mixed code state</h3>{raw_quote(samples['e1_mix'])}</section>
      <section class="card"><h3>Exp003 explicit parent recovery</h3>{raw_quote(samples['e3_explicit'])}</section>
      <section class="card"><h3>Exp004 midpoint</h3>{raw_quote(samples['e4_mid'])}</section>
      <section class="card"><h3>Exp007 stochastic midpoint</h3>{raw_quote(samples['e7_mid'])}</section>
      <section class="card"><h3>Exp008 numerical midpoint</h3>{raw_quote(samples['e8_mid'])}</section>
      <section class="card"><h3>Exp008 direct prompt</h3>{raw_quote(samples['e8_direct'])}</section>
      <section class="card"><h3>Exp009 numerical midpoint</h3>{raw_quote(samples['e9_mid'])}</section>
      <section class="card"><h3>Exp009 direct prompt</h3>{raw_quote(samples['e9_direct'])}</section>
      <section class="card"><h3>Exp011 matrix endpoint A</h3>{raw_quote(samples['e11_a'])}</section>
      <section class="card"><h3>Exp011 matrix endpoint B</h3>{raw_quote(samples['e11_b'])}</section>
      <section class="card"><h3>Exp012 numerical midpoint</h3>{raw_quote(samples['e12_mid'])}</section>
      <section class="card"><h3>Exp012 direct matrix prompt</h3>{raw_quote(samples['e12_direct'])}</section>
      <section class="card"><h3>Exp012 graph strict-gate failure</h3>{raw_quote(samples['e12_graph_damaged'])}</section>
      <section class="card"><h3>Exp013 full-context graph success</h3>{raw_quote(samples['e13_graph'])}</section>
      <section class="card"><h3>Exp013 image strict-gate failure</h3>{raw_quote(samples['e13_image_strict_failure'])}</section>
    </div>"""

    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Technical report — diffusion idea composition pilot</title>
<style>{CSS}@media print{{header{{background:{NAVY}!important;-webkit-print-color-adjust:exact}} main{{max-width:none}}}}</style></head>
<body><header><h1>Technical Report: Diffusion Representation Composition Pilot</h1>
<p class="lede">Experiments 001–013 are mandatory and read directly from raw manifests and text. Build date: {BUILD_DATE}.</p></header><main>
<section><h2>Executive summary</h2><div class="callout"><p class="result">Endpoint-valid aggregate: {c['controlled_numerical'][0]}/{c['controlled_numerical'][1]} numerical hybrids versus {c['controlled_direct'][0]}/{c['controlled_direct'][1]} strict direct-prompt hybrids.</p>
<p>The bounded pilot result now spans sorting, dense matrix multiplication, sparse graph BFS, and image normalization. It does <strong>not</strong> show that all diffusion methods fail. It shows that the tested final-canvas arithmetic interpolation, hard projection, stochastic top-k projection, and per-step linear output-logit mixing did not provide a useful semantic composition coordinate.</p></div></section>
<section><h2>Full-context blank-canvas follow-up</h2><p>Experiment 013 implemented Sergiy's follow-up suggestion by placing both complete frozen parent records in one prompt and supplying no parent-derived canvas state. It produced {c['exp13_full'][0]}/{c['exp13_full'][1]} strict hybrids, exactly tying Experiment 012's concise direct-prompt baseline at {c['exp13_concise'][0]}/{c['exp13_concise'][1]}. Matrix remained 3/3, graph improved to 3/3, and image declined to 2/3 because one output omitted order preservation.</p>
<p>The ninth output still combined fixed-size batching, memory capping, NumPy broadcasting, and float32, but the preregistered strict outcome remains a failure. The follow-up supports prompt-only blank-canvas synthesis as a viable baseline; it does not improve aggregate reliability or rehabilitate numerical mixing.</p></section>
<section><h2>Research question and operational definition</h2><p>Can prompt-conditioned numerical representations from two optimization ideas be combined so DiffusionGemma generates one coherent idea retaining both parents?</p>
<p>A candidate hybrid had to satisfy the experiment-specific lexical groups for Parent A and Parent B and avoid prohibited operations. Endpoint gates prevented interpretation when the model could not first reproduce both parents.</p></section>
<section><h2>System and provenance</h2><p>Where recorded, the full-model runs used <code>google/diffusiongemma-26B-A4B-it</code>, model revisions {h(', '.join(model_revisions) or 'not consistently recorded')}, and an NVIDIA GeForce RTX 5090. Exact settings, seeds, hashes, timestamps, and runtime details remain in each raw manifest.</p>
<p>The partial-denoising and autoregressive handoff status comes from an earlier unnumbered September 16 pilot, read at build time from: <code>{h(c['prior_pilot_sources'][0])}</code>, <code>{h(c['prior_pilot_sources'][1])}</code>, <code>{h(c['prior_pilot_sources'][2])}</code>, and <code>{h(c['prior_pilot_sources'][3])}</code>.</p>
<table><thead><tr><th>Exp</th><th>Directory</th><th>Status</th><th>Repository revision</th><th>Model</th><th>Manifest read at build</th></tr></thead><tbody>{provenance_rows(e)}</tbody></table></section>
<section><h2>Experiment ladder</h2><div class="grid">{experiment_cards(e,c)}</div></section>
<section><h2>Aggregate plots</h2><div class="grid">
<section class="card"><img class="plot" src="{plots['exp004'].relative_to(HERE).as_posix()}" alt="Experiment 004 categories by interpolation weight."><p>The sweep directly counts raw <code>retains_A</code>, <code>retains_B</code>, and <code>candidate_hybrid</code> fields.</p></section>
<section class="card"><img class="plot" src="{plots['exp005'].relative_to(HERE).as_posix()}" alt="Experiment 005 geometry metrics."><p>Positive cosine and retained norm argue against cancellation.</p></section>
<section class="card"><img class="plot" src="{plots['exp006'].relative_to(HERE).as_posix()}" alt="Experiment 006 top-k endpoint token availability."><p>Both alternatives often exist below top-1, but hard independent projection cannot preserve sequence-level meaning.</p></section>
<section class="card"><img class="plot" src="{plots['methods'].relative_to(HERE).as_posix()}" alt="Hybrid rate by tested method."><p>Numerical methods produced {c['controlled_numerical'][0]}/{c['controlled_numerical'][1]} strict hybrids while direct prompts produced {c['controlled_direct'][0]}/{c['controlled_direct'][1]} across four workloads.</p></section>
</div></section>
{sample_sections}
<section><h2>Sergiy’s explicit proposal items and their status</h2><p>This table attributes the original requested directions separately from later ideas developed by the project team.</p>
<table><thead><tr><th>Explicitly proposed/requested item</th><th>Evidence-based status</th></tr></thead><tbody>
<tr><td>Mix prompt/internal representations from blank output canvases.</td><td><strong>Tested negative for the implemented final-state methods.</strong> Endpoints decoded, but final-state interpolation/projection and linear output-logit mixing produced {c['controlled_numerical'][0]}/{c['controlled_numerical'][1]} strict hybrids across endpoint-valid pairs.</td></tr>
<tr><td>Use Darwin optimization ideas and programs.</td><td><strong>Partly executed.</strong> Darwin-derived optimization ideas drove the sorting studies, and code-state controls were attempted, but no mixed output established a useful optimized program or Darwin fitness gain.</td></tr>
<tr><td>Stop denoising early, then ask a larger autoregressive model to complete the scaffold.</td><td><strong>Tested in the earlier pilot; no useful information was added.</strong> {c['partial_scaffold_useful']}/{c['partial_scaffold_runs']} partial outputs were useful. Scaffold-only handoff recovered neither parent in {c['handoff_scaffold_only']}; parents plus scaffold succeeded {c['handoff_parents_and_scaffold']}, but parents-only had already succeeded {c['handoff_parents_only']}.</td></tr>
<tr><td>Use Bayesian optimization in representation space.</td><td><strong>Not justified.</strong> The tested coordinate crosses broad unrelated regions rather than a smooth, useful semantic bridge.</td></tr>
<tr><td>Eventually integrate a synthesis/search stage into Darwin for code optimization.</td><td><strong>Not justified for the failed numerical mechanism.</strong> Direct text-based idea synthesis remains viable as a simpler candidate, but program-level Darwin implementation and benchmarking remain untested.</td></tr>
</tbody></table></section>
<section><h2>Our later future-work ideas—not Sergiy’s explicit asks</h2><ul>
<li>Internal transformer-layer injection.</li>
<li>Prompt KV-cache mixing.</li>
<li>Learned sequence-aware mixing or steering.</li>
<li>A preregistered confirmatory study with held-out pairs, blinded evaluation, and program benchmarks.</li>
</ul></section>
<section><h2>What was not completed—and why</h2><p>These were downstream implementation aspirations in the proposal, contingent on the research mechanism first passing scientific gates. Their omission is a documented decision, not forgotten work.</p><ul>
<li><strong>No production Darwin sampling-stage integration:</strong> the tested numerical synthesis mechanism showed no benefit, so production integration was not scientifically justified.</li>
<li><strong>No Bayesian optimizer run:</strong> the smooth-coordinate gate failed; the tested path crossed a broad unrelated region rather than a useful continuous space.</li>
<li><strong>No end-to-end generated-program compilation, test, and benchmark campaign:</strong> the numerical method produced no useful synthesized idea to advance, and the initial code controls were invalid.</li>
</ul><p>The underlying questions—whether the tested representations compose, whether partial handoff helps, whether optimization is justified, and whether this mechanism merits Darwin integration—were nevertheless addressed.</p></section>
<section><h2>Limitations and scope</h2><ul>
<li>All results are exploratory/pilot and were not preregistered confirmatory evidence.</li>
<li>The strongest comparison spans endpoint-valid sorting, matrix multiplication, graph BFS, and image-normalization pairs; Experiment 009's histogram pair and Experiment 011's regex pair were blocked by endpoint controls.</li>
<li>Experiment 012's graph seed 43 was a strict automated-gate failure because <code>indptr</code> was damaged to <code>` `ptr</code>, although the intended deque-plus-CSR combination was human-identifiable.</li>
<li>Experiment 013 tied rather than exceeded the concise prompt baseline. Its image seed 44 output combined the core vectorization and batching techniques but omitted the frozen output-order requirement.</li>
<li>Automated lexical indicators are transparent and reproducible but do not replace blinded human evaluation, code implementation, tests, benchmarks, or uncertainty estimates.</li>
<li>The conclusion is bounded to the exact model revision, representations, projection/injection points, prompts, seeds, canvases, and workloads tested.</li>
<li>Internal transformer-layer injection, KV-cache mixing, learned sequence-aware methods, and preregistered confirmation are later project-proposed future work, not Sergiy’s explicit asks. The negative result is specifically bounded to final-state interpolation/projection and linear output-logit mixing.</li>
</ul></section>
<section><h2>Our recommended future work</h2><p>These are later project recommendations, not additional items attributed to Sergiy’s original proposal.</p><ol>
<li>Freeze a paper-grade confirmatory protocol, held-out pairs, exclusions, outcome rules, run counts, and blinded evaluation.</li>
<li>Use the sorting and nonsorting results to define a multi-workload confirmatory corpus without converting these pilot runs into confirmatory evidence.</li>
<li>Test a technically distinct sequence-aware mechanism: internal-layer or KV-cache intervention with preserved token alignment.</li>
<li>Keep direct diffusion prompting and autoregressive combination as strong baselines.</li>
<li>Only revisit Bayesian optimization after demonstrating a smooth, controllable, useful low-dimensional coordinate.</li>
</ol></section></main><footer>Generated by <code>deliverables/build.py</code>. Raw experiment files were read without modification.</footer></body></html>"""
    REPORT_PATH.write_text(document, encoding="utf-8")


def build_answers(e: dict[int, dict[str, Any]], c: dict[str, Any]) -> None:
    text = f"""# Answers to Sergiy’s proposal

## Attribution key

The first five sections below are items **explicitly requested or proposed by Sergiy/the submission**. The later section titled **“Our proposed future work”** contains research directions developed after the original proposal and must not be attributed to Sergiy.

## 1. Prompt/internal representation mixing from blank canvases — explicit proposal item

**Status: tested negative for the implemented final-state mechanisms.** The experiments began with blank output canvases and mixed prompt-conditioned numerical states or output logits. DiffusionGemma decoded validated endpoints and generated fluent text, but fluency was not faithful composition. Experiment 004 produced **{c['exp4_hybrids']}/{c['exp4_intermediate_count']}** hybrids at intermediate weights. Across endpoint-valid Experiments 008, 009, and 012, final-state projection and linear output-logit mixing produced **{c['controlled_numerical'][0]}/{c['controlled_numerical'][1]}** strict hybrids.

This result is bounded to final-state interpolation/projection and linear output-logit mixing. It does not test every possible internal representation intervention.

## 2. Darwin ideas and programs — explicit proposal item

**Status: partly executed.** Darwin-derived optimization ideas supplied the initial sorting parents, and Experiment 001 attempted code-state controls. However, all Experiment 001 generated programs were invalid Python, and no numerical mixture established a useful program or a Darwin benchmark improvement. The later nonsorting pairs were public synthetic optimization tasks used to test whether the mechanism generalized beyond sorting.

## 3. Partial denoising followed by larger autoregressive completion — explicit proposal item

**Status: tested in the earlier pilot; it added no useful information.** The partial-denoising study tested **{c['partial_scaffold_runs']}** midpoint outputs and found **{c['partial_scaffold_useful']}** useful partial scaffolds. In the explicit handoff, scaffold-only completions recovered neither parent in **{c['handoff_scaffold_only']}** cases. Parents plus scaffold succeeded **{c['handoff_parents_and_scaffold']}**, but the same parents without the scaffold had already succeeded **{c['handoff_parents_only']}**. The larger autoregressive model succeeded because it received the parent ideas, not because the partial diffusion output preserved a useful combination.

## 4. Bayesian optimization in representation space — explicit proposal item

**Status: not justified for the tested coordinate.** Experiment 004’s endpoint-controlled sweep moved from an A-like basin through a broad unrelated region to a B-like basin, with no hybrid region. The aggregate numerical result was **{c['controlled_numerical'][0]}/{c['controlled_numerical'][1]}** hybrids. Bayesian optimization would therefore search a discontinuous transition rather than a demonstrated smooth, useful idea space.

## 5. Eventual Darwin search-stage integration and code optimization — explicit proposal item

**Status: not justified for the failed numerical mechanism.** It would add complexity and GPU cost without a demonstrated synthesis benefit. A simpler direct text-based idea synthesizer remains viable: direct diffusion prompting produced **{c['controlled_direct'][0]}/{c['controlled_direct'][1]}** strict hybrids across endpoint-valid sorting, matrix, graph, and image pairs. Program implementation, correctness testing, benchmarking, and fitness improvement inside Darwin remain untested.

## Follow-up suggested by Sergiy on September 18: everything in the prompt, blank canvas

**Status: tested successfully, with no aggregate improvement over the concise prompt.** Experiment 013 placed both complete frozen parent records in one prompt and supplied no parent embedding, hidden state, token canvas, or logits. Full-context prompting produced **{c['exp13_full'][0]}/{c['exp13_full'][1]}** strict hybrids, exactly tying Experiment 012's concise blank-canvas baseline at **{c['exp13_concise'][0]}/{c['exp13_concise'][1]}**.

Matrix multiplication remained 3/3. Graph BFS improved from 2/3 to 3/3, while image normalization declined from 3/3 to 2/3 because one output omitted the frozen output-order requirement. Human inspection found the core two-technique combination in that ninth output, but the preregistered primary result remains 8/9. These public synthetic tasks contain complete parent descriptions rather than complete parent programs, so a full-program-context study remains technically distinct.

## What was not completed—and why

These were **implementation aspirations from the proposal**, contingent on the research mechanism first passing scientific gates. They were not forgotten:

- **No production Darwin sampling-stage integration.** The tested numerical synthesis mechanism had no demonstrated benefit, so integration was not scientifically justified.
- **No Bayesian optimizer run.** The required smooth-coordinate gate failed: interpolation crossed a broad unrelated region rather than a useful continuous idea space.
- **No end-to-end generated-program compilation, test, and benchmark campaign.** The numerical mechanism produced no useful synthesized idea to advance, and the initial code-state outputs were invalid.

The underlying research questions were still addressed: the tested representation mixing was negative; partial scaffold/handoff added no useful information; Bayesian optimization was not justified; and the failed mechanism did not merit Darwin integration. Direct text-based idea synthesis remains a viable separate path.

## What succeeded?

- Reproducible full-model execution with recorded prompts, hashes, seeds, revisions, runtime, and raw outputs.
- Parent endpoint decoding when prompts were specific enough.
- A diagnostic explanation stronger than “the vectors cancelled”: global cosine was **{c['geometry']['global_cosine']:.3f}**, no positions had negative cosine, and midpoint norm retention was **{c['geometry']['midpoint_norm_retention_vs_mean_parent_norm']:.1%}**.
- Evidence that parent alternatives often remain below top-1, identifying independent vocabulary projection as a likely bottleneck.
- Direct diffusion prompting composed both parents in **{c['controlled_direct'][0]}/{c['controlled_direct'][1]}** strict runs across four workloads.
- Experiment 011 found **{c['exp11_passing']}/{c['exp11_pair_count']}** endpoint-valid nonsorting pairs: matrix multiplication, graph BFS, and image normalization; regex was blocked.
- Experiment 012 replicated the numerical failure on all three endpoint-valid nonsorting pairs. Matrix direct prompting passed 3/3, graph 2/3 strict, and image 3/3.
- Experiment 013 tested Sergiy's full-context blank-canvas follow-up and tied the concise baseline at {c['exp13_full'][0]}/{c['exp13_full'][1]} strict hybrids. Graph improved to 3/3 while image declined to 2/3.
- The remaining graph direct output was human-identifiable as deque plus CSR, but a damaged `indptr` token correctly made it a strict automated-gate failure.

## Our proposed future work — not Sergiy’s explicit asks

These directions were developed later by the project team:

- Internal transformer-layer injection and layer selection.
- Prompt KV-cache mixing.
- Learned sequence-aware or alignment-aware mixing.
- A preregistered confirmatory study with held-out pairs, multiple workloads, blinded judges, uncertainty estimates, and program benchmarks.

They are technically distinct future studies, not unfinished versions of Sergiy’s original final-state mixing experiment.

## Scope statement

These are **exploratory/pilot** findings for the exact DiffusionGemma revision, prompts, final-canvas states, linear output-logit mixing, projection methods, weights, seeds, and workloads tested. They do **not** establish that all diffusion language-model composition methods fail. The negative result is bounded to final-state interpolation/projection and linear output-logit mixing.

## Provenance

- Numbered experiment evidence: `experiments/001-*` through `experiments/013-*`.
- Earlier partial-scaffold and handoff evidence: `{c['prior_pilot_sources'][0]}`, `{c['prior_pilot_sources'][1]}`, `{c['prior_pilot_sources'][2]}`, and `{c['prior_pilot_sources'][3]}`.
"""
    ANSWERS_PATH.write_text(text, encoding="utf-8")


def rgb(hex_color: str) -> RGBColor:
    value = hex_color.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def add_alt_text(shape: Any, description: str) -> None:
    try:
        shape._element._nvXxPr.cNvPr.set("descr", description)
    except AttributeError:
        pass


def set_background(slide: Any, color: str = LIGHT) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)


def add_text(
    slide: Any,
    text: str,
    x: float,
    y: float,
    w: float,
    hgt: float,
    size: int = 24,
    color: str = DARK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(hgt))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = rgb(color)
    return box


def add_bullets(slide: Any, bullets: Iterable[str], x: float, y: float, w: float, hgt: float, size: int = 21) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(hgt))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for index, text in enumerate(bullets):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = text
        p.level = 0
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = rgb(DARK)
        p.space_after = Pt(10)
    return box


def add_title(slide: Any, title: str, kicker: str | None = None) -> None:
    add_text(slide, title, 0.65, 0.35, 12.0, 0.65, 29, NAVY, True)
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.65), Inches(1.08), Inches(1.5), Inches(0.07))
    line.fill.solid()
    line.fill.fore_color.rgb = rgb(PURPLE)
    line.line.fill.background()
    if kicker:
        add_text(slide, kicker, 2.35, 0.92, 10.2, 0.35, 12, MID)


def add_footer(slide: Any, source: str) -> None:
    add_text(slide, f"Pilot evidence • {BUILD_DATE} • {source}", 0.55, 7.12, 12.2, 0.22, 9, MID)


def add_image(slide: Any, path: Path, x: float, y: float, w: float, hgt: float, alt: str) -> Any:
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(hgt))
    add_alt_text(pic, alt)
    return pic


def add_card(slide: Any, x: float, y: float, w: float, hgt: float, title: str, body: str, accent: str = BLUE) -> None:
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(hgt))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(WHITE)
    shape.line.color.rgb = rgb("#D8E1EE")
    stripe = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(0.1), Inches(hgt))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = rgb(accent)
    stripe.line.fill.background()
    add_text(slide, title, x + 0.22, y + 0.16, w - 0.35, 0.42, 17, NAVY, True)
    add_text(slide, body, x + 0.22, y + 0.65, w - 0.38, hgt - 0.8, 14, DARK)


def add_notes_footer(slide: Any, source_paths: list[str]) -> None:
    add_footer(slide, " | ".join(source_paths))


def new_slide(prs: Presentation, title: str, kicker: str | None = None) -> Any:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    add_title(slide, title, kicker)
    return slide


def build_pptx(
    e: dict[int, dict[str, Any]],
    c: dict[str, Any],
    plots: dict[str, Path],
    alt: dict[str, str],
    samples: dict[str, tuple[str, str]],
) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide, NAVY)
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.16))
    band.fill.solid(); band.fill.fore_color.rgb = rgb(CYAN); band.line.fill.background()
    add_text(slide, "Can AI Average Two\nOptimization Ideas?", 0.8, 1.0, 7.6, 2.0, 38, WHITE, True)
    add_text(slide, "A controlled pilot study of DiffusionGemma representations for Project Darwin", 0.85, 3.15, 7.5, 1.0, 22, SKY)
    add_text(slide, "Research communication package • September 18, 2026", 0.85, 5.9, 8.0, 0.4, 14, WHITE)
    for i, (label, color) in enumerate([("IDEA A", BLUE), ("MIX", PURPLE), ("IDEA B", CYAN)]):
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(9.1), Inches(1.25 + i * 1.55), Inches(2.7), Inches(0.9))
        shape.fill.solid(); shape.fill.fore_color.rgb = rgb(color); shape.line.fill.background()
        add_text(slide, label, 9.1, 1.25 + i * 1.55, 2.7, 0.9, 19, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)

    slide = new_slide(prs, "The accessible version: what are we trying to do?")
    add_card(slide, 0.75, 1.45, 3.75, 4.6, "1 • Start with two useful ideas", "Example A: use a bounded counting sort.\n\nExample B: compile custom loops with Numba.", BLUE)
    add_card(slide, 4.8, 1.45, 3.75, 4.6, "2 • Combine their numbers", "A language model represents text using large arrays of numbers. We tested whether averaging or mixing those arrays preserves both meanings.", PURPLE)
    add_card(slide, 8.85, 1.45, 3.75, 4.6, "3 • Ask for one new idea", "Success means the generated idea is coherent, relevant, and clearly uses both parent techniques—not merely fluent text.", CYAN)
    add_notes_footer(slide, ["research design summarized from experiments/001–013/spec.json"])

    slide = new_slide(prs, "How diffusion text generation differs", "Simple conceptual model—not a claim about every implementation detail")
    stages = [("Prompt", "describes the task"), ("Continuous state", "numbers carry contextual information"), ("Token projection", "each position becomes a token choice"), ("Denoising", "tokens are refined into text")]
    for i, (title, body) in enumerate(stages):
        x = 0.65 + i * 3.15
        add_card(slide, x, 2.0, 2.65, 2.5, title, body, PALETTE[i])
        if i < len(stages) - 1:
            add_text(slide, "→", x + 2.7, 2.75, 0.4, 0.6, 28, PURPLE, True, PP_ALIGN.CENTER)
    add_text(slide, "The key risk: averaging can preserve numbers without preserving a valid sequence-level meaning.", 1.1, 5.25, 11.2, 0.75, 23, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/005-projection-geometry-diagnostic", "experiments/006-top-candidate-projection"])

    slide = new_slide(prs, "Research question, hypothesis, and success rule")
    add_card(slide, 0.8, 1.4, 5.8, 2.0, "Research question", "Can prompt-conditioned numerical representations from two optimization ideas be combined so the model generates one coherent idea retaining both parents?", BLUE)
    add_card(slide, 6.9, 1.4, 5.6, 2.0, "Hypothesis", "Careful alignment plus interpolation or logit mixing may create a controllable path between parent concepts.", PURPLE)
    add_card(slide, 0.8, 3.75, 5.8, 2.15, "Success", "Both parent indicator groups present; coherent task-relevant idea; no prohibited shortcut.", GREEN)
    add_card(slide, 6.9, 3.75, 5.6, 2.15, "Control gate", "Do not interpret a midpoint unless both endpoints first reproduce their own parent concepts.", AMBER)
    add_notes_footer(slide, ["experiments/004/spec.json", "experiments/009/spec.json"])

    slide = new_slide(prs, "Controlled experimental pipeline")
    pipeline = [
        ("Freeze parents", "named techniques"),
        ("Align prompts", "same layout"),
        ("Validate endpoints", "A and B controls"),
        ("Manipulate", "states / tokens / logits"),
        ("Generate", "fixed seeds + steps"),
        ("Score", "transparent indicators"),
    ]
    for i, (title, body) in enumerate(pipeline):
        x = 0.45 + i * 2.12
        add_card(slide, x, 2.1, 1.82, 2.4, title, body, PALETTE[i % len(PALETTE)])
        if i < 5:
            add_text(slide, "→", x + 1.82, 2.95, 0.3, 0.45, 18, MID, True, PP_ALIGN.CENTER)
    add_text(slide, "Raw outputs, failures, hashes, seeds, model revision, and runtime are preserved in each experiment directory.", 0.85, 5.25, 11.7, 0.8, 20, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/001–013/results/results.json"])

    slide = new_slide(prs, "Twelve experiments formed the diagnostic ladder", "Experiment 013 then tested Sergiy's prompt-only follow-up")
    labels = [
        "001 Code controls", "002 Prompt controls", "003 Specificity", "004 Weight sweep",
        "005 Geometry", "006 Top candidates", "007 Stochastic", "008 Dual logits",
        "009 Sorting replication", "010 Endpoint repair", "011 Nonsorting screen", "012 Nonsorting composition",
    ]
    for i, label in enumerate(labels):
        row, col = divmod(i, 4)
        x, y = 0.55 + col * 3.16, 1.35 + row * 1.65
        add_card(slide, x, y, 2.85, 1.25, label, "completed", GREEN)
    add_text(slide, "013 follow-up: complete parent records in one prompt, with no parent-derived canvas state.", 0.8, 6.25, 11.7, 0.5, 16, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/001–013"])

    slide = new_slide(prs, "Experiments 001–003: controls changed the question")
    add_card(slide, 0.65, 1.4, 3.85, 4.8, "001 • Code states", "Parent techniques were recognizable, but all generated programs—including the mixture—were invalid Python.\n\nLesson: move first to shorter idea generation.", BLUE)
    add_card(slide, 4.75, 1.4, 3.85, 4.8, "002 • Prompt states", "The model retained generic counting sort but lost NumPy, bincount, and repeat from Parent B.\n\nLesson: the midpoint was uninterpretable because an endpoint was weak.", PURPLE)
    add_card(slide, 8.85, 1.4, 3.85, 4.8, "003 • Specificity", "An explicit source prompt recovered NumPy + bincount + repeat from 1 through 8 steps.\n\nLesson: prompt specificity is a causal control, not cosmetic wording.", CYAN)
    add_notes_footer(slide, ["experiments/001/results/results.json", "experiments/002/results/results.json", "experiments/003/results/results.json"])

    slide = new_slide(prs, "Experiment 004: a controlled sweep found no semantic bridge")
    add_image(slide, plots["exp004"], 0.65, 1.35, 8.0, 4.75, alt["exp004-weight-regions.png"])
    add_card(slide, 8.95, 1.55, 3.65, 1.25, "Endpoints", "Both parents passed across 3 seeds.", GREEN)
    add_card(slide, 8.95, 3.0, 3.65, 1.25, "Intermediate weights", f"{c['exp4_hybrids']}/{c['exp4_intermediate_count']} hybrids.", RED)
    add_card(slide, 8.95, 4.45, 3.65, 1.65, "Interpretation", "Categorical A-like, unrelated, and B-like regions—not a smooth compositional path.", PURPLE)
    add_notes_footer(slide, ["experiments/004-controlled-prompt-weight-sweep/results/results.json"])

    slide = new_slide(prs, "Experiment 005: vector cancellation was not the explanation")
    add_image(slide, plots["exp005"], 0.7, 1.45, 7.8, 4.55, alt["exp005-geometry.png"])
    add_card(slide, 8.8, 1.55, 3.8, 1.25, "Positive similarity", f"Global cosine = {c['geometry']['global_cosine']:.3f}.", BLUE)
    add_card(slide, 8.8, 3.0, 3.8, 1.25, "No opposite positions", f"Negative position count = {c['geometry']['position_cosine']['negative_position_count']}.", CYAN)
    add_card(slide, 8.8, 4.45, 3.8, 1.55, "Strong retained magnitude", f"Midpoint retained {c['geometry']['midpoint_norm_retention_vs_mean_parent_norm']:.1%} of mean parent norm.", PURPLE)
    add_notes_footer(slide, ["experiments/005-projection-geometry-diagnostic/results/results.json"])

    slide = new_slide(prs, "Experiment 006: the hard projection creates a patchwork")
    add_image(slide, plots["exp006"], 0.65, 1.45, 7.8, 4.55, alt["exp006-topk-availability.png"])
    add_card(slide, 8.75, 1.55, 3.9, 1.4, "Top-1", "No endpoint-distinct position had both alternatives available at rank 1.", RED)
    add_card(slide, 8.75, 3.15, 3.9, 1.4, "Top-20", f"Both alternatives appeared at {c['exp6_availability']['20']['both_present_fraction']:.1%} of positions.", GREEN)
    add_card(slide, 8.75, 4.75, 3.9, 1.4, "Likely bottleneck", "Independent per-position argmax discards sequence-level combinations.", PURPLE)
    add_notes_footer(slide, ["experiments/006-top-candidate-projection/results/results.json"])

    slide = new_slide(prs, "Experiment 007: stochastic top-20 sampling did not rescue meaning")
    add_text(slide, f"{c['exp7_hybrids']} / {c['exp7_midpoints']}", 0.8, 1.55, 4.4, 1.5, 48, RED, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    add_text(slide, "midpoint hybrids", 0.8, 2.95, 4.4, 0.6, 22, NAVY, True, PP_ALIGN.CENTER)
    add_card(slide, 5.55, 1.5, 6.5, 1.35, "Controls", "Parent A and B endpoints passed at temperatures 0.5, 1.0, and 2.0.", GREEN)
    add_card(slide, 5.55, 3.1, 6.5, 1.35, "Intervention", "Sample independently from temperature-scaled top-20 token candidates, then denoise for 8 steps.", BLUE)
    add_card(slide, 5.55, 4.7, 6.5, 1.35, "Result", "More token variety did not reconstruct a coherent combination of counting sort and Numba.", PURPLE)
    add_notes_footer(slide, ["experiments/007-stochastic-projection/results/results.json"])

    slide = new_slide(prs, "Experiment 008: mixing logits each denoising step still failed")
    add_image(slide, plots["methods"], 0.55, 1.4, 7.5, 4.7, alt["method-comparison.png"])
    add_card(slide, 8.35, 1.5, 4.3, 1.2, "Dual-logit intermediates", f"{c['exp8_dual'][0]}/{c['exp8_dual'][1]} hybrids.", RED)
    add_card(slide, 8.35, 2.9, 4.3, 1.2, "Hard midpoint", f"{c['exp8_hard'][0]}/{c['exp8_hard'][1]} hybrids.", RED)
    add_card(slide, 8.35, 4.3, 4.3, 1.2, "Direct diffusion prompt", f"{c['exp8_direct'][0]}/{c['exp8_direct'][1]} hybrids.", GREEN)
    add_text(slide, "The model could combine the ideas when told explicitly, but not through the tested numerical paths.", 8.45, 5.75, 4.0, 0.55, 15, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/008-dual-prompt-logit-mixing/results/results.json"])

    slide = new_slide(prs, "Experiment 009: replication across pairs")
    add_card(slide, 0.7, 1.45, 3.8, 4.7, "Vectorized counting", "Endpoint gate: PASS\n\nNumerical midpoints: 0/6\nDirect prompts: 3/3", BLUE)
    add_card(slide, 4.75, 1.45, 3.8, 4.7, "Fast path + fallback", "Endpoint gate: PASS\n\nNumerical midpoints: 0/6\nDirect prompts: 3/3", PURPLE)
    add_card(slide, 8.8, 1.45, 3.8, 4.7, "Parallel histogram", "Endpoint gate: BLOCKED\n\nParent B failed 2/3 seeds. No midpoint inference was allowed.", AMBER)
    add_notes_footer(slide, ["experiments/009-multi-pair-replication/results/results.json"])

    slide = new_slide(prs, "Experiment 010: reproducibility passed; endpoint specificity did not")
    add_card(slide, 0.8, 1.45, 3.7, 4.7, "Deterministic reproduction", f"{c['exp10_reproduction'][0]}/{c['exp10_reproduction'][1]} raw outputs reproduced canonically.", GREEN)
    add_card(slide, 4.8, 1.45, 3.7, 4.7, "Strengthened Parent A", f"{c['exp10_strong_a'][0]}/{c['exp10_strong_a'][1]} seeds retained all required groups.", AMBER)
    add_card(slide, 8.8, 1.45, 3.7, 4.7, "Strengthened Parent B", f"{c['exp10_strong_b'][0]}/{c['exp10_strong_b'][1]} seeds retained all required groups.\n\nTherefore no midpoint was admissible.", RED)
    add_notes_footer(slide, ["experiments/010-histogram-endpoint-specificity/results/results.json"])

    slide = new_slide(prs, "Experiment 011: nonsorting endpoint screen")
    add_text(slide, f"{c['exp11_passing']} / {c['exp11_pair_count']}", 0.7, 1.55, 3.6, 1.3, 46, GREEN, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    add_text(slide, "pairs passed both endpoints", 0.7, 2.85, 3.6, 0.6, 20, NAVY, True, PP_ALIGN.CENTER)
    add_card(slide, 4.65, 1.45, 3.65, 4.7, "Passed", "Matrix: BLAS + tiling\n\nGraph: deque BFS + CSR\n\nImage: vectorization + batching", BLUE)
    add_card(slide, 8.65, 1.45, 3.65, 4.7, "Blocked", "Regex + memory mapping\n\nCompiled-regex endpoint passed only 1/3 seeds, so the pair was excluded from midpoint testing.", AMBER)
    add_notes_footer(slide, ["experiments/011-nonsorting-endpoint-screen/results/results.json"])

    slide = new_slide(prs, "Experiment 012: the negative result replicated beyond sorting")
    add_card(slide, 0.55, 1.4, 3.85, 4.85, "Matrix multiplication", f"Dual logit: 0/3\nHard midpoint: 0/3\nDirect prompt: {c['exp12_direct_by_pair']['matrix-blocked-blas'][0]}/3 strict\n\nDirect prompts combined BLAS-backed matmul with cache-sized blocks.", BLUE)
    add_card(slide, 4.74, 1.4, 3.85, 4.85, "Sparse graph BFS", f"Dual logit: 0/3\nHard midpoint: 0/3\nDirect prompt: {c['exp12_direct_by_pair']['graph-csr-deque-bfs'][0]}/3 strict\n\nThe third was human-identifiable but damaged `indptr` to `` ` `ptr``.", PURPLE)
    add_card(slide, 8.93, 1.4, 3.85, 4.85, "Image normalization", f"Dual logit: 0/3\nHard midpoint: 0/3\nDirect prompt: {c['exp12_direct_by_pair']['image-batched-vectorization'][0]}/3 strict\n\nDirect prompts combined float32 broadcasting with bounded batches.", CYAN)
    add_notes_footer(slide, ["experiments/012-nonsorting-composition/results/results.json", "experiments/012-nonsorting-composition/results/assessment.md"])

    slide = new_slide(prs, "Experiment 013: all parent context in the prompt, blank canvas")
    add_card(slide, 0.55, 1.4, 3.85, 4.7, "Prompt-only configuration", "Both complete frozen parent records were placed in one prompt.\n\nNo parent embedding, hidden state, token canvas, or logits were supplied.", BLUE)
    add_card(slide, 4.74, 1.4, 3.85, 4.7, "Strict aggregate", f"Full context: {c['exp13_full'][0]}/{c['exp13_full'][1]}\nConcise baseline: {c['exp13_concise'][0]}/{c['exp13_concise'][1]}\n\nAggregate improvement: none.", GREEN)
    add_card(slide, 8.93, 1.4, 3.85, 4.7, "Pair-level shift", f"Matrix: {c['exp13_full_by_pair']['matrix-blocked-blas'][0]}/3\nGraph: {c['exp13_full_by_pair']['graph-csr-deque-bfs'][0]}/3\nImage: {c['exp13_full_by_pair']['image-batched-vectorization'][0]}/3\n\nThe image failure omitted output order but retained vectorization and batching.", PURPLE)
    add_text(slide, "Result: Sergiy's blank-canvas prompt-only approach works reliably, but the longer context did not outperform the concise explicit prompt.", 0.8, 6.25, 11.7, 0.55, 18, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/013-full-context-blank-canvas/results/results.json", "experiments/013-full-context-blank-canvas/results/assessment.md"])

    slide = new_slide(prs, "Aggregate endpoint-valid result across four workloads")
    add_image(slide, plots["aggregate"], 0.65, 1.4, 7.1, 4.8, alt["controlled-aggregate.png"])
    add_card(slide, 8.1, 1.55, 4.35, 1.6, "Numerical representation methods", f"{c['controlled_numerical'][0]}/{c['controlled_numerical'][1]} candidate hybrids.", RED)
    add_card(slide, 8.1, 3.45, 4.35, 1.6, "Direct diffusion prompting", f"{c['controlled_direct'][0]}/{c['controlled_direct'][1]} strict candidate hybrids.", GREEN)
    add_text(slide, "Pilot contrast—not a population estimate.", 8.2, 5.55, 4.1, 0.5, 16, MID, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/008/results/results.json", "experiments/009/results/results.json", "experiments/012/results/results.json"])

    slide = new_slide(prs, "What actually succeeded?")
    add_bullets(
        slide,
        [
            "Reproducible full-model execution with seeds, hashes, revisions, timings, and preserved raw outputs.",
            "Endpoint decoding when prompts named the required implementation details.",
            "A causal diagnostic: poor composition was not explained by simple vector cancellation.",
            "A likely bottleneck: useful endpoint alternatives survived below top-1 but hard projection broke sequence-level meaning.",
            f"Direct diffusion prompting composed both parents {c['controlled_direct'][0]}/{c['controlled_direct'][1]} strict times across sorting, matrix, graph, and image pairs.",
            f"Full-context blank-canvas prompting independently achieved {c['exp13_full'][0]}/{c['exp13_full'][1]}, tying rather than improving the concise prompt.",
            "Endpoint gates prevented overinterpreting the blocked histogram and regex pairs.",
        ],
        1.0, 1.4, 11.4, 5.4, 21,
    )
    add_notes_footer(slide, ["experiments/003–012/results/results.json"])

    slide = new_slide(prs, "Sergiy’s explicit proposal: status of each item")
    add_card(slide, 0.45, 1.3, 3.8, 2.0, "1 • Mix prompt/internal states", f"Blank-canvas final-state and output-logit methods were tested. Result: {c['controlled_numerical'][0]}/{c['controlled_numerical'][1]} strict numerical hybrids.", RED)
    add_card(slide, 4.76, 1.3, 3.8, 2.0, "2 • Darwin ideas/programs", "Darwin-derived ideas were used; code-state controls were invalid. No Darwin program fitness gain was established.", AMBER)
    add_card(slide, 9.07, 1.3, 3.8, 2.0, "3 • Partial denoise → larger AR", f"Earlier pilot: {c['partial_scaffold_useful']}/{c['partial_scaffold_runs']} useful scaffolds; scaffold-only handoff recovered neither parent in {c['handoff_scaffold_only']}. It added no useful information.", RED)
    add_card(slide, 1.25, 3.65, 5.25, 1.85, "4 • Bayesian optimization", "Not justified: the tested coordinate crosses a broad unrelated region rather than a smooth useful bridge.", RED)
    add_card(slide, 6.83, 3.65, 5.25, 1.85, "5 • Darwin search-stage integration", "Not justified for the failed numerical mixer. Direct text-based idea synthesis remains viable; program-level Darwin evaluation is still untested.", PURPLE)
    add_text(slide, "Attribution boundary: internal-layer injection, KV-cache mixing, learned sequence-aware mixing, and preregistered confirmation are our later future-work ideas—not Sergiy’s explicit asks.", 0.65, 5.95, 12.0, 0.75, 16, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/001–013", "earlier pilot: partial-scaffolds and scaffold-handoff assessments", "answers-to-sergiy.md"])

    slide = new_slide(prs, "What was not completed—and why", "Implementation aspirations were gated by the research result")
    add_card(slide, 0.65, 1.45, 3.8, 4.35, "No production Darwin stage", "The tested numerical synthesis mechanism showed no benefit. Building it into the production sampling pipeline was therefore not scientifically justified.", PURPLE)
    add_card(slide, 4.77, 1.45, 3.8, 4.35, "No Bayesian optimizer run", "The smooth-coordinate gate failed: the path crossed a broad unrelated region instead of a useful continuous semantic space.", RED)
    add_card(slide, 8.89, 1.45, 3.8, 4.35, "No program campaign", "No end-to-end generated-program compilation, test, and benchmark campaign followed because the numerical method produced no useful synthesized idea to advance.", AMBER)
    add_text(slide, "These were proposal implementation aspirations, not forgotten tasks. The underlying research questions were answered, and stopping at the failed gates was the scientifically justified outcome.", 0.75, 6.05, 11.85, 0.7, 17, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/001/results/results.json", "experiments/004–012/results/results.json", "answers-to-sergiy.md"])

    slide = new_slide(prs, "Limitations: what this study cannot claim")
    add_bullets(
        slide,
        [
            "Exploratory/pilot evidence; not preregistered confirmatory evidence.",
            "The strongest composition comparison covers four workloads but one DiffusionGemma revision and public synthetic tasks.",
            "Lexical indicators are reproducible but do not replace blinded human judgment or program benchmarks.",
            "The graph 2/3 strict result includes a human-identifiable third output whose `indptr` token was damaged.",
            "Negative findings apply to final-state interpolation/projection and linear output-logit mixing—not all diffusion methods.",
            "Our later future-work ideas—not Sergiy’s explicit asks—include internal-layer injection, KV-cache mixing, learned sequence-aware mixing, and preregistered confirmation.",
        ],
        0.95, 1.4, 11.5, 5.5, 22,
    )
    add_notes_footer(slide, ["technical-report.html § Limitations and scope"])

    slide = new_slide(prs, "Our future-work plan—not Sergiy’s explicit asks")
    add_card(slide, 0.65, 1.35, 3.8, 4.9, "1 • Preregister confirmation", "Freeze hypotheses, held-out pairs, outcomes, exclusions, seeds, run counts, blinded evaluation, and program benchmarks.", BLUE)
    add_card(slide, 4.77, 1.35, 3.8, 4.9, "2 • Test distinct mechanisms", "Try internal-layer injection, KV-cache mixing, or learned sequence-aware mixing that preserves sequence structure.", PURPLE)
    add_card(slide, 8.89, 1.35, 3.8, 4.9, "3 • Keep the viable baseline", f"Direct text-based synthesis is simpler and succeeded in {c['controlled_direct'][0]}/{c['controlled_direct'][1]} strict runs across four workloads. Revisit optimization only after finding a smooth useful coordinate.", GREEN)
    add_text(slide, "Conclusion: the tested numerical bridge failed, but the diagnostic ladder produced a clear, reproducible, and scientifically useful negative result.", 0.85, 6.35, 11.7, 0.55, 18, NAVY, True, PP_ALIGN.CENTER)
    add_notes_footer(slide, ["experiments/001–013", "answers-to-sergiy.md"])

    require(18 <= len(prs.slides) <= 23, f"Presentation must contain 18–23 slides; generated {len(prs.slides)}")
    prs.save(PPTX_PATH)


def verify_outputs() -> None:
    expected = [PPTX_PATH, POSTER_PATH, REPORT_PATH, ANSWERS_PATH, *ASSETS.glob("*.png")]
    require(len(list(ASSETS.glob("*.png"))) >= 5, "Expected at least five generated plots")
    for path in expected:
        require(path.exists() and path.stat().st_size > 0, f"Generated output is empty or missing: {path}")
    reopened = Presentation(PPTX_PATH)
    require(18 <= len(reopened.slides) <= 23, f"Reopened PPTX has unexpected slide count: {len(reopened.slides)}")
    require(len(reopened.slides) == 23, f"Expected 23 slides after reopening; found {len(reopened.slides)}")


def main() -> int:
    try:
        evidence = load_evidence()
        prior_scaffold_evidence = load_prior_scaffold_evidence()
        claims = calculate_claims(evidence, prior_scaffold_evidence)
        samples = collect_samples(evidence)
        plots, alt = build_plots(claims)
        build_answers(evidence, claims)
        build_poster(evidence, claims, plots)
        build_report(evidence, claims, plots, samples)
        build_pptx(evidence, claims, plots, alt, samples)
        verify_outputs()
    except EvidenceError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"Built and verified communication package in {HERE}")
    print(f"PPTX slides: {len(Presentation(PPTX_PATH).slides)}")
    for path in [PPTX_PATH, POSTER_PATH, REPORT_PATH, ANSWERS_PATH, *sorted(ASSETS.glob('*.png'))]:
        print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
