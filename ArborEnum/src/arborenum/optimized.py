from . import ArborEnum
from ._core_opt import ArborEnum as _ArborEnumCoreOpt

__all__ = ["ArborEnumOptimized"]
class ArborEnumOptimized(ArborEnum):
    def __init__(self):
        super().__init__()
        self._model = _ArborEnumCoreOpt()

    def equiv_stats(self):
        """Counters for equivalent split grouping (cumulative across fits).

        evals:  thresholds whose proxy objectives were computed individually
        reused: thresholds that reused an equivalent threshold's child subgraphs
                (each one is a proxy evaluation + subgraph solve that was skipped)
        """
        evals = int(self._model.equiv_threshold_evals)
        reused = int(self._model.equiv_threshold_reused)
        total = evals + reused
        return {
            "evals": evals,
            "reused": reused,
            "reuse_fraction": (reused / total) if total else 0.0,
        }

    def reset_equiv_stats(self):
        self._model.reset_equiv_stats()
