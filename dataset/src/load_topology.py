from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from .utils import sanitize_attrs, write_json


def load_topology(path: Path, directed: bool = False) -> nx.Graph:
    graph = nx.read_graphml(path)
    if not directed:
        graph = graph.to_undirected()

    # Normalize node labels to strings for consistent downstream processing.
    graph = nx.relabel_nodes(graph, lambda n: str(n))

    # Ensure we do not carry self-loops that could break flow/path generation.
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def export_topology(graph: nx.Graph, path: Path, name: str) -> None:
    nodes: List[Dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({"id": str(node_id), **sanitize_attrs(attrs)})

    edges: List[Dict[str, Any]] = []
    for u, v, attrs in graph.edges(data=True):
        edges.append({"u": str(u), "v": str(v), **sanitize_attrs(attrs)})

    payload: Dict[str, Any] = {
        "name": name,
        "directed": graph.is_directed(),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
    }
    write_json(path, payload)
