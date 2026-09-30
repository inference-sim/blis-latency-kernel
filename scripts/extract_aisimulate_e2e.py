#!/usr/bin/env python3
"""Extract the scoreable per-point corpus from NVIDIA AISimulate's e2e accuracy artifact.

The artifact is published by the `E2E Accuracy Matrix` workflow in ai-dynamo/aisimulate as
`e2e-accuracy-web`, and carries a nested summary:

    models -> workloads -> gpus -> topologies -> points

Each point holds `measured`, `aic` and `aisimulate` blocks of TTFT/TPOT relatives. A topology
is one deployment at one workload; its points are a concurrency sweep.

# Provenance, stated because it is mixed

The predictions and the snapshot are NVIDIA's. The `measured_*` ground truth is NOT: the
artifact's own `measurement_source` is "SemiAnalysis InferenceX", with a release URL. This
project's REGISTRY is single-sourced from AISimulate and contains no InferenceX data -- that
constraint is about coefficients, the numbers the model is built from. This file is an
evaluation set, a different role: it is what the model is scored against, never fitted to. The
distinction is recorded here so a later reader does not mistake one for the other.

# Why relatives, and what that costs

Absolute latencies are not disclosed, so every value is normalised to its sweep's lowest
concurrency. That is why no simulator is needed to score it -- a ratio of step times at one
fixed deployment is a pure function of the kernel -- and also why it measures shape only.

# Selection

A topology is emitted when it has at least three successful points. Two points carry a single
ratio, which pins a level rather than a curvature -- scoring a shape against it measures almost
nothing and dilutes the reported figure with sweeps that cannot disagree. Aggregated serving with no speculation is the whole of the reachable set
here; both are asserted rather than assumed, so a future artifact that adds disaggregated or
speculative arms fails loudly instead of being scored under the wrong deployment.

Usage:

    python scripts/extract_aisimulate_e2e.py summary.json -o testdata/measurements/aisimulate_e2e.json
"""

import argparse
import collections
import json
import sys

# Artifact model slug -> blis-catalog model name. A slug absent here is skipped: the catalog
# has no graph for it, so it cannot be priced.
MODELS = {
    "gptoss120b": "gpt-oss-120b",
    "minimaxm2.5": "minimax-m2.5",
    "qwen3.5": "qwen3.5-397b-a17b",
    "glm5": "glm-5",
}

# Artifact GPU slug -> blis-catalog hardware name.
CHIPS = {"h100": "h100", "h200": "h200", "b200": "b200", "b300": "b300"}

MIN_POINTS = 3


def scenario_name(model: str, gpu: str, precision: str, framework: str, par: dict) -> str:
    """One file per distinct deployment, named from the fields that define it."""
    ep = par["moe_ep_size"]
    parts = [model, gpu, precision, framework, f"tp{par['tp_size']}"]
    if ep > 1:
        parts.append(f"ep{ep}")
    if par["attention_dp_size"] > 1:
        parts.append(f"dp{par['attention_dp_size']}")
    return "-".join(parts) + ".yaml"


def extract(summary: dict) -> dict:
    sweeps = []
    skipped = collections.Counter()
    for model in summary["models"]:
        name = MODELS.get(model["model"])
        if name is None:
            skipped["model not in catalog"] += 1
            continue
        for workload in model["workloads"]:
            for gpu in workload["gpus"]:
                chip = CHIPS.get(gpu["gpu"])
                if chip is None:
                    skipped["gpu not in catalog"] += 1
                    continue
                for topology in gpu["topologies"]:
                    serving = topology["serving"]
                    spec = topology.get("spec_method") or "none"
                    if serving != "aggregated" or spec != "none":
                        # Not reachable by a step-time ratio at one deployment.
                        skipped[f"serving={serving} spec={spec}"] += 1
                        continue
                    par = topology["parallelism"]
                    if par["pp_size"] != 1:
                        skipped["pp_size>1"] += 1
                        continue
                    points = [
                        {
                            "concurrency": p["concurrency"],
                            "measured_tpot_relative": p["measured"]["tpot_relative"],
                            "aisimulate_tpot_relative": p["aisimulate"]["tpot_relative"],
                            "aisimulate_tpot_error_pct": p["aisimulate"]["tpot_error_pct"],
                            "status": p["status"],
                        }
                        for p in topology["points"]
                        if p.get("status") == "success"
                    ]
                    if len(points) < MIN_POINTS:
                        skipped["fewer than three successful points"] += 1
                        continue
                    points.sort(key=lambda p: p["concurrency"])
                    sweeps.append(
                        {
                            "scenario": scenario_name(
                                name, chip, topology["precision"],
                                topology["framework"], par,
                            ),
                            "model": name,
                            "gpu": chip,
                            "workload": workload["identity"],
                            "label": workload["label"],
                            "framework": topology["framework"],
                            "precision": topology["precision"],
                            "serving": serving,
                            "spec_method": spec,
                            "parallelism": par,
                            "topology_id": topology["id"],
                            "points": points,
                        }
                    )
    sweeps.sort(key=lambda s: (s["model"], s["gpu"], s["scenario"], s["label"]))
    return {
        "source": "NVIDIA AISimulate e2e accuracy artifact (e2e-accuracy-web)",
        "measurement_source": summary["snapshot"].get("measurement_source"),
        "measurement_source_url": summary["snapshot"].get("measurement_source_url"),
        "snapshot": summary["snapshot"],
        "scope": summary["scope"],
        # Carries both tpot figures. tpot_shape_error_pct is the one comparable to a
        # normalised-curve score; tpot_mape_pct is absolute.
        "aisimulate_totals": summary["totals"].get("aisimulate", {}),
        "selection": {
            "models": sorted(set(MODELS.values())),
            "min_points_per_sweep": MIN_POINTS,
            "serving": "aggregated",
            "spec_method": "none",
            "skipped": dict(skipped),
        },
        "sweeps": sweeps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="summary.json from the e2e-accuracy-web artifact")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    with open(args.summary) as fh:
        summary = json.load(fh)
    corpus = extract(summary)

    scenarios = {s["scenario"] for s in corpus["sweeps"]}
    points = sum(len(s["points"]) for s in corpus["sweeps"])
    with open(args.out, "w") as fh:
        json.dump(corpus, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"{len(corpus['sweeps'])} sweeps, {points} points, "
          f"{len(scenarios)} distinct deployments -> {args.out}")
    for reason, n in sorted(corpus["selection"]["skipped"].items()):
        print(f"  skipped {n:4d}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
