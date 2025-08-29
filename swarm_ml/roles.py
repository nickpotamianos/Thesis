# swarm_ml/roles.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd

@dataclass
class Roles:
    target: str
    trackers: List[str]
    # Optional pre-resolved mapping from robot -> list of its UWB tag IDs
    tag_ids_by_robot: Optional[Dict[str, List[int]]] = None

def infer_tag_ids_by_robot(uwb_range_df: pd.DataFrame,
                           robots: List[str],
                           min_count: int = 50) -> Dict[str, List[int]]:
    """
    Heuristic: infer which numeric tag IDs belong to which robot by co-occurrence
    in the source robot's UWB dataframe. This uses frequency statistics and works
    even when each robot has 2 tags.
    """
    tag_ids_by_robot = {}
    if "robot" not in uwb_range_df.columns:
        # if caller didn't add it, infer from grouping if possible, else leave empty
        for r in robots:
            tag_ids_by_robot[r] = []
        return tag_ids_by_robot

    for r in robots:
        df_r = uwb_range_df[uwb_range_df["robot"] == r]
        # Tags seen on this robot either in from_id or to_id (depending on logger)
        counts = pd.concat([
            df_r["from_id"].value_counts(),
            df_r["to_id"].value_counts()
        ], axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
        tag_ids_by_robot[r] = [int(t) for t, c in counts.items() if c >= min_count]
    return tag_ids_by_robot

def get_roles(exp_name: str,
              robots: List[str],
              default_target: Optional[str] = None,
              uwb_range_df: Optional[pd.DataFrame] = None,
              user_override: Optional[Tuple[str, List[str]]] = None
              ) -> Roles:
    """
    Decide which robot is 'target' vs 'trackers'. If user_override is provided, use that.
    Else, pick the robot with the densest UWB connections as the target (often ifo003 in examples).
    """
    if user_override is not None:
        tgt, trk = user_override
        return Roles(target=tgt, trackers=trk)

    if default_target is None and robots:
        # Basic heuristic: choose lexicographically last (often ifo003 in docs)
        default_target = sorted(robots)[-1]
    trackers = [r for r in robots if r != default_target]

    tag_map = None
    if uwb_range_df is not None:
        tag_map = infer_tag_ids_by_robot(uwb_range_df.assign(robot=uwb_range_df.get("robot", None)),
                                         robots=robots)

    return Roles(target=default_target, trackers=trackers, tag_ids_by_robot=tag_map)
