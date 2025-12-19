import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple
from utils.data import FlowDataset, load_paths
from utils.eval import evaluate_assignment, evaluate_cover


def greedy_set_cover(dataset: FlowDataset) -> Tuple[List[int], Set[int]]:
    """Standard greedy set cover: pick switch covering most uncovered flows."""
    switch_to_flows: List[Set[int]] = [set() for _ in range(dataset.n_switches)]
    for f_idx, switches in enumerate(dataset.P):
        for sid in switches:
            switch_to_flows[sid].add(f_idx)

    uncovered: Set[int] = set(range(dataset.n_flows))
    selected: List[int] = []

    while uncovered:
        best_sid = None
        best_gain = 0
        for sid, flows in enumerate(switch_to_flows):
            gain = len(uncovered.intersection(flows))
            if gain > best_gain:
                best_gain = gain
                best_sid = sid
        if best_sid is None or best_gain == 0:
            break
        selected.append(best_sid)
        uncovered.difference_update(switch_to_flows[best_sid])

    return selected, uncovered


def assign_flows(dataset: FlowDataset, selected: List[int]) -> Dict[int, int]:
    """Assign each flow to the first selected switch on its path."""
    selected_set = set(selected)
    assignments: Dict[int, int] = {}
    for f_idx, switches in enumerate(dataset.P):
        for sid in switches:
            if sid in selected_set:
                assignments[f_idx] = sid
                break
    return assignments

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy set cover for flow paths.")
    parser.add_argument("--input", required=True, help="Path to flow paths JSONL (.gz ok).")
    parser.add_argument("--out-dir", help="Output directory (default: out/run_<timestamp>).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_paths(Path(args.input))
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_greedy(dataset, out_dir)


def run_greedy(dataset: FlowDataset, out_dir: Path) -> None:
    selected_ids, uncovered = greedy_set_cover(dataset)
    assignments = assign_flows(dataset, selected_ids)

    cover_ok, uncovered_list = evaluate_cover(dataset, selected_ids)
    assignment_eval = evaluate_assignment(dataset, assignments)

    solution = {
        "status": "Greedy",
        "objective": len(selected_ids),
        "selected_switch_ids": selected_ids,
        "selected_switch_names": [dataset.sid_to_name[sid] for sid in selected_ids],
        "assignments_by_id": assignments,
        "assignments": {dataset.fid_to_name[f]: dataset.sid_to_name[s] for f, s in assignments.items()},
        "cover_ok": cover_ok,
        "uncovered_flows": list(uncovered),
        **assignment_eval,
    }

    solution_path = out_dir / "solution.json"
    with solution_path.open("w", encoding="utf-8") as fh:
        json.dump(solution, fh, indent=2, sort_keys=True)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Flows={dataset.n_flows} | Switches={dataset.n_switches}\n")
        fh.write("Model: greedy\n")
        fh.write(f"Selected switches: {len(selected_ids)}\n")
        fh.write(f"Coverage ok: {cover_ok}\n")
        fh.write(f"Uncovered flows: {len(uncovered_list)}\n")

    print(f"[done] Greedy selected {len(selected_ids)} switches | out={out_dir}")


def _default_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"run_{ts}"


if __name__ == "__main__":
    main()
