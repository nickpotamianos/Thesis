# swarm_ml/los_adapter.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import inspect, os, sys, importlib.util

def _load_from_fs(fname="los_classification.py") -> Optional[object]:
    """Load LOS module from explicit path (LOS_MODULE_PATH) or search repo recursively."""
    root = os.getcwd()
    # 1) Honor explicit environment variable
    try:
        env_path = os.environ.get("LOS_MODULE_PATH", None)
        if env_path and os.path.exists(env_path):
            spec = importlib.util.spec_from_file_location("los_classification", env_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            print(f"[LOS] Loaded {fname} from explicit path: {env_path}", flush=True)
            return mod
    except Exception as e:
        print(f"[LOS] Failed to load LOS module from LOS_MODULE_PATH: {e}", flush=True)
    for r, _, files in os.walk(root):
        if fname in files:
            path = os.path.join(r, fname)
            spec = importlib.util.spec_from_file_location("los_classification", path)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                print(f"[LOS] Loaded {fname} from: {path}", flush=True)
                return mod
            except Exception as e:
                print(f"[LOS] Failed to load {path}: {e}", flush=True)
                return None
    print(f"[LOS] {fname} not found under: {root}", flush=True)
    return None

try:
    # Prefer a local copy under swarm_ml if you duplicated it
    from swarm_ml import los_classification as losmod  # type: ignore
except Exception:
    try:
        import los_classification as losmod  # repo root
    except Exception:
        losmod = _load_from_fs()  # <- NEW: auto-import from file if present

@dataclass
class LOSConfig:
    use_cir: bool = False
    default_score: float = 0.5
    los_score_if_true: float = 0.9
    los_score_if_false: float = 0.1
    verbose: bool = True
    per_row_average: bool = True

class LOSAdapter:
    """Auto-detect a LOS function in los_classification.py and return a score in [0,1]."""
    def __init__(self, cfg: LOSConfig = LOSConfig()):
        self.cfg = cfg
        self.enabled = losmod is not None
        self._printed = False
        self._cand_funcs: List[str] = []
        self._row_funcs: List[str] = []
        self._chosen: Optional[str] = None
        if self.enabled:
            self._introspect()

    def _print_once(self, msg: str):
        if self.cfg.verbose and not self._printed:
            print(msg, flush=True)

    def _introspect(self):
        names = dir(losmod)
        funcs = [n for n in names if inspect.isfunction(getattr(losmod, n))]
        self._cand_funcs = [n for n in funcs if any(k in n.lower() for k in
                           ["predict_los_probability","predict_los_prob","predict_proba",
                            "los_probability","los_prob","classify","score","predict"])]
        self._row_funcs = [n for n in funcs if any(k in n.lower() for k in
                           ["predict_row","classify_row","los_row"])]
        self._chosen = (self._cand_funcs[0] if self._cand_funcs
                        else self._row_funcs[0] if self._row_funcs else None)
        if self.cfg.verbose:
            self._print_once("[LOS] los_classification module found.")
            self._print_once(f"[LOS] Functions: {funcs}")
            self._print_once(f"[LOS] Batch candidates: {self._cand_funcs}, Row candidates: {self._row_funcs}")
            self._print_once(f"[LOS] Selected callable: {self._chosen}")
            self._printed = True

    @staticmethod
    def _as_score(out: Any, cfg: LOSConfig) -> Optional[float]:
        try:
            if out is None: return None
            if isinstance(out, (float, int)):
                v = float(out)
                return float(np.clip(v,0,1)) if 0<=v<=1 else (cfg.los_score_if_true if v!=0 else cfg.los_score_if_false)
            if isinstance(out, bool):
                return cfg.los_score_if_true if out else cfg.los_score_if_false
            if isinstance(out, dict):
                for k in ("prob_los","los_prob","p_los","score","los"):
                    if k in out:
                        v = out[k]
                        if isinstance(v, bool): return cfg.los_score_if_true if v else cfg.los_score_if_false
                        return float(np.clip(float(v),0,1))
            if hasattr(out, "__len__"):
                arr = np.asarray(out, dtype=float).reshape(-1)
                return float(np.clip(arr,0,1).mean())
        except Exception:
            return None
        return None

    def score(self, pair_df: pd.DataFrame, extras: Optional[Dict[str, Any]] = None) -> Optional[float]:
        if not self.enabled or pair_df is None or pair_df.empty or self._chosen is None:
            return None
        func = getattr(losmod, self._chosen, None)
        if func is None: return None

        # Try DataFrame
        try:
            s = self._as_score(func(pair_df), self.cfg)
            if s is not None: return s
        except Exception as e:
            self._print_once(f"[LOS] Batch call {self._chosen}(DataFrame) failed: {e}")

        # Try ranges
        try:
            s = self._as_score(func(pair_df["range"].to_numpy(dtype=float)), self.cfg)
            if s is not None: return s
        except Exception as e:
            self._print_once(f"[LOS] Batch call {self._chosen}(ranges) failed: {e}")

        # Try row-wise average
        if self.cfg.per_row_average:
            scores=[]
            for _, row in pair_df.iterrows():
                for cand in self._row_funcs + self._cand_funcs:
                    f = getattr(losmod, cand, None)
                    if f is None: continue
                    try:
                        s = self._as_score(f(row.to_dict()), self.cfg)
                        if s is not None: scores.append(s); break
                    except Exception: continue
            if scores: return float(np.clip(np.mean(scores),0,1))

        # Fallback if no classifier available
        if pair_df is not None and not pair_df.empty:
            zs = pair_df["range"].to_numpy(dtype=float)
            if zs.size >= 4:
                q75, q25 = np.percentile(zs, 75), np.percentile(zs, 25)
                iqr = max(1e-3, q75 - q25)
                # Map smaller IQR to higher LOS score in [0.3, 0.8]
                score = 0.8 - 0.5 * float(np.clip(iqr, 0.0, 1.0))
                return score
            else:
                # With few samples, be cautiously neutral
                return 0.5
        return None
