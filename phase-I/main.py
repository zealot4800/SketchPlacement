import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pulp

from ilp.ilp import build_set_cover_model, extract_switch_selection, solve_model
from utils.data import FlowDataset, load_paths
from utils.eval import evaluate_cover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow ILP: set cover model.")
    parser.add_argument("--input", required=True, help="Path to flow paths JSONL (.gz ok).")
    parser.add_argument("--out-dir", help="Output directory (default: out/run_<timestamp>).")
    parser.add_argument("--skip-solve", action="store_true", help="Write LP but skip solving.")
    parser.add_argument("--mode", choices=["solve", "preprocess", "eval"], default="solve")
    parser.add_argument("--solution", help="Path to solution.json for eval mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if args.mode == "preprocess":
        dataset = load_paths(input_path)
        print(_summary(dataset))
        return

    if args.mode == "eval":
        if not args.solution:
            raise ValueError("--solution required for eval mode")
        dataset = load_paths(input_path)
        run_eval(dataset, Path(args.solution))
        return

    # Solve mode
    dataset = load_paths(input_path)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_solve(args, dataset, out_dir)


def run_solve(args: argparse.Namespace, dataset: FlowDataset, out_dir: Path) -> None:
    model_name = "cover"
    model, x_vars = build_set_cover_model(dataset)
    y_vars = None

    lp_path = out_dir / "model.lp"
    result_status = None
    if args.skip_solve:
        model.writeLP(str(lp_path))
    else:
        result_status = solve_model(model, write_lp=str(lp_path))

    solution: Dict[str, Any] = {
        "status": pulp.LpStatus[result_status] if result_status is not None else "NotSolved",
        "objective": float(pulp.value(model.objective)) if result_status == pulp.LpStatusOptimal else None,
    }

    selected_ids = [sid for sid, var in x_vars.items() if result_status is not None and pulp.value(var) > 0.5]
    selected_names = [dataset.sid_to_name[sid] for sid in selected_ids]
    solution["selected_switch_names"] = selected_names
    solution["selected_switch_ids"] = selected_ids

    solution_path = out_dir / "solution.json"
    with solution_path.open("w", encoding="utf-8") as fh:
        json.dump(solution, fh, indent=2, sort_keys=True)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(_summary(dataset))
        fh.write("\n")
        fh.write(f"Model: {model_name}\n")
        fh.write(f"Status: {solution['status']}\n")
        fh.write(f"Objective: {solution['objective']}\n")
        if result_status is not None:
            fh.write(f"Selected switches: {len(selected_names)}\n")
        else:
            fh.write("Model not solved (skip-solve enabled).\n")

    selected_count = len(selected_ids) if result_status is not None else 0
    print(f"[done] Selected {selected_count} switches | Status={solution['status']} | out={out_dir}")


def run_eval(dataset: FlowDataset, solution_path: Path) -> None:
    with solution_path.open("r", encoding="utf-8") as fh:
        sol = json.load(fh)

    selected_ids = sol.get("selected_switch_ids", [])
    selected_names = sol.get("selected_switch_names", [])
    if not selected_ids and selected_names:
        name_to_id = {name: idx for idx, name in enumerate(dataset.sid_to_name)}
        selected_ids = [name_to_id[n] for n in selected_names if n in name_to_id]

    ok_cover, uncovered = evaluate_cover(dataset, selected_ids)
    print(json.dumps({"cover_ok": ok_cover, "uncovered_flows": uncovered}, indent=2))


def _summary(dataset: FlowDataset) -> str:
    cap_info = "none"
    if dataset.capacities is not None:
        positive = int(np.count_nonzero(dataset.capacities))
        cap_info = f"{positive} switches with caps"
    return f"Flows={dataset.n_flows} | Switches={dataset.n_switches} | Caps={cap_info}"


def _default_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"run_{ts}"


if __name__ == "__main__":
    main()
