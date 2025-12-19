from typing import Dict, List, Tuple
from pulp import LpVariable

import numpy as np
import pulp

from utils.data import FlowDataset


def build_set_cover_model(dataset: FlowDataset) -> Tuple[pulp.LpProblem, Dict[int, LpVariable]]:
    model = pulp.LpProblem("set_cover_switches", pulp.LpMinimize)
    x_vars: Dict[int, LpVariable] = {
        sid: pulp.LpVariable(f"x_{sid}", lowBound=0, upBound=1, cat="Binary") for sid in range(dataset.n_switches)
    }

    for f_idx, switches in enumerate(dataset.P):
        model += pulp.lpSum(x_vars[sid] for sid in switches) >= 1, f"cover_flow_{f_idx}"

    model += pulp.lpSum(x_vars[sid] for sid in range(dataset.n_switches))
    return model, x_vars


def build_assignment_model(
    dataset: FlowDataset,
    lambda_penalty: float = 0.0,
    capacities: np.ndarray | None = None,
) -> Tuple[pulp.LpProblem, Dict[int, LpVariable], Dict[Tuple[int, int], LpVariable]]:
    
    model = pulp.LpProblem("switch_assignment", pulp.LpMinimize)
    x_vars: Dict[int, LpVariable] = {
        sid: pulp.LpVariable(f"x_{sid}", lowBound=0, upBound=1, cat="Binary") for sid in range(dataset.n_switches)
    }
    y_vars: Dict[Tuple[int, int], LpVariable] = {}

    for f_idx, switches in enumerate(dataset.P):
        flow_y = []
        for sid in switches:
            var = pulp.LpVariable(f"y_{f_idx}_{sid}", lowBound=0, upBound=1, cat="Binary")
            y_vars[(f_idx, sid)] = var
            flow_y.append(var)
            model += var <= x_vars[sid], f"assign_implies_select_f{f_idx}_s{sid}"
        model += pulp.lpSum(flow_y) == 1, f"assign_once_f{f_idx}"

    cap_array = capacities if capacities is not None else dataset.capacities
    if cap_array is not None:
        if len(cap_array) != dataset.n_switches:
            raise ValueError("Capacity array length must match number of switches.")
        for sid in range(dataset.n_switches):
            cap_val = float(cap_array[sid])
            if cap_val > 0:
                model += (
                    pulp.lpSum(y_vars[(f_idx, sid)] for f_idx, _ in _flows_covering_sid(dataset, sid, y_vars))
                    <= cap_val * x_vars[sid],
                    f"capacity_s{sid}",
                )

    model += pulp.lpSum(x_vars.values()) + lambda_penalty * pulp.lpSum(y_vars.values())
    return model, x_vars, y_vars


def _flows_covering_sid(dataset: FlowDataset, sid: int, y_vars: Dict[Tuple[int, int], LpVariable]):
    for f_idx, switches in enumerate(dataset.P):
        if sid in switches and (f_idx, sid) in y_vars:
            yield (f_idx, switches)


def solve_model(model: pulp.LpProblem, write_lp: str | None = None) -> int:
    if write_lp:
        model.writeLP(write_lp)
    solver = pulp.PULP_CBC_CMD(msg=False)
    result_status = model.solve(solver)
    return result_status


def extract_switch_selection(x_vars: Dict[int, LpVariable], sid_to_name: List[str]) -> List[str]:
    selected = []
    for sid, var in x_vars.items():
        if pulp.value(var) and pulp.value(var) > 0.5:
            selected.append(sid_to_name[sid])
    return selected


def extract_assignments(
    y_vars: Dict[Tuple[int, int], LpVariable], fid_to_name: List[str], sid_to_name: List[str]
) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    for (f_idx, sid), var in y_vars.items():
        if pulp.value(var) and pulp.value(var) > 0.5:
            assignments[fid_to_name[f_idx]] = sid_to_name[sid]
    return assignments


def extract_assignments_idx(y_vars: Dict[Tuple[int, int], LpVariable]) -> Dict[int, int]:
    assignments: Dict[int, int] = {}
    for (f_idx, sid), var in y_vars.items():
        if pulp.value(var) and pulp.value(var) > 0.5:
            assignments[f_idx] = sid
    return assignments
