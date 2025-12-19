from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


Flow = Dict[str, object]


def _gravity_probabilities(graph: nx.Graph, allow_self_flows: bool) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    nodes = list(graph.nodes())
    degrees = dict(graph.degree())
    pairs: List[Tuple[str, str]] = []
    weights: List[float] = []

    for src in nodes:
        for dst in nodes:
            if not allow_self_flows and src == dst:
                continue
            weight = (degrees.get(src, 0) + 1) * (degrees.get(dst, 0) + 1)
            if weight > 0:
                pairs.append((src, dst))
                weights.append(float(weight))

    if not pairs:
        raise ValueError("No valid source/destination pairs found for flow generation.")

    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    return pairs, probs


def generate_flows(
    graph: nx.Graph,
    count: int,
    model: str = "gravity",
    demand_scale: float = 10.0,
    demand_sigma: float = 0.8,
    allow_self_flows: bool = False,
) -> List[Flow]:
    if model != "gravity":
        raise ValueError(f"Unsupported flow model: {model}")

    pairs, probs = _gravity_probabilities(graph, allow_self_flows)
    indices = np.random.choice(len(pairs), size=count, p=probs)

    flows: List[Flow] = []
    for flow_id, pair_idx in enumerate(indices):
        src, dst = pairs[pair_idx]
        demand = float(np.random.lognormal(mean=np.log(demand_scale), sigma=demand_sigma))
        flows.append({"id": int(flow_id), "src": src, "dst": dst, "demand": demand})
    return flows


def write_flows_csv(flows: Sequence[Flow], path) -> None:
    import csv
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "src", "dst", "demand"])
        writer.writeheader()
        for flow in flows:
            writer.writerow(
                {
                    "id": flow["id"],
                    "src": str(flow["src"]),
                    "dst": str(flow["dst"]),
                    "demand": f"{float(flow['demand']):.6f}",
                }
            )
