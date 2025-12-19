import gzip
import json
import logging
from typing import Dict, Iterable, List, Optional, Sequence

import networkx as nx


PathRecord = Dict[str, object]


def compute_shortest_paths(
    graph: nx.Graph, flows: Sequence[Dict[str, object]], weight_attr: Optional[str] = None
) -> List[PathRecord]:
    paths: List[PathRecord] = []
    skipped = 0

    for flow in flows:
        src = flow["src"]
        dst = flow["dst"]
        try:
            path_nodes = nx.shortest_path(graph, source=src, target=dst, weight=weight_attr)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped += 1
            continue

        total_weight = 0.0
        if weight_attr:
            # Sum edge weights if present; fallback to hop count otherwise.
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = graph.get_edge_data(u, v, default={})
                total_weight += float(edge_data.get(weight_attr, 1.0))
        else:
            total_weight = float(len(path_nodes) - 1)

        paths.append(
            {
                "id": flow["id"],
                "src": str(src),
                "dst": str(dst),
                "path": [str(node) for node in path_nodes],
                "hops": len(path_nodes) - 1,
                "cost": total_weight,
            }
        )

    if skipped:
        logging.warning("Skipped %d flows without a valid path", skipped)
    return paths


def write_paths_jsonl_gz(paths: Iterable[PathRecord], path) -> None:
    out_path = str(path)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for record in paths:
            fh.write(json.dumps(record))
            fh.write("\n")
