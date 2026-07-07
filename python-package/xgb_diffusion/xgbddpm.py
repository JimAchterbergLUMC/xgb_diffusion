"""XGBDDPM estimators with per-boosting-round diffusion noise."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from ._data_utils import array_interface
from .callback import CallbackContainer, EarlyStopping, EvaluationMonitor, TrainingCallback
from .config import config_context
from .core import Booster, DMatrix, _LIB, _check_call, make_jcargs
from .sklearn import XGBClassifier, XGBModel, XGBRegressor


class _XGBDDPMMixin:
    _ddpm_params = {
        "alpha_bars",
        "n_classes",
        "noise_samples_per_row",
        "timestep",
        "target_index",
        "ddpm_objective",
        "refresh_every_k",
        "min_boosting_round",
    }

    def _wrapper_params(self) -> set:
        return super()._wrapper_params() | self._ddpm_params

    def _check_ddpm_params(self) -> None:
        if self.alpha_bars is None:
            raise ValueError("alpha_bars is required.")
        if self.target_index is None:
            raise ValueError("target_index is required.")

    def _empty_dmatrix(self, X: np.ndarray) -> DMatrix:
        rows = X.shape[0] * int(self.noise_samples_per_row)
        cols = X.shape[1]
        if self.timestep is None:
            rows *= len(self.alpha_bars)
            cols += 1
        return DMatrix(
            np.zeros((rows, cols), dtype=np.float32),
            label=np.zeros(rows, dtype=np.float32),
            nthread=self.n_jobs,
            missing=self.missing,
            feature_types=self.feature_types,
        )

    def _refresh_dmatrix(
        self,
        dmat: DMatrix,
        X: np.ndarray,
        y: np.ndarray,
        iteration: int,
        validation: bool = False,
    ) -> None:
        objective = {"x": 0, "eps": 1, "v": 2}[self.ddpm_objective]
        refresh_every_k = int(self.refresh_every_k)
        seed = int(self.random_state or 0) + (10_000_000 if validation else 0)
        if refresh_every_k >= 1:
            seed += iteration // refresh_every_k
        config = make_jcargs(
            noise_samples_per_row=int(self.noise_samples_per_row),
            timestep=-1 if self.timestep is None else int(self.timestep),
            target_index=int(self.target_index),
            objective=objective,
            seed=seed,
            n_classes=[] if self.n_classes is None else list(map(int, self.n_classes)),
        )
        _check_call(
            _LIB.XGDMatrixXGBDDPMRefresh(
                dmat.handle,
                array_interface(X),
                array_interface(y),
                array_interface(self.alpha_bars),
                config,
            )
        )

    def _fit_ddpm(
        self,
        X: Any,
        y: Any,
        *,
        eval_set: Optional[Sequence[Tuple[Any, Any]]] = None,
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        feature_weights: Optional[Any] = None,
    ) -> "_XGBDDPMMixin":
        self._check_ddpm_params()
        if callable(self.objective):
            raise NotImplementedError(
                "Callable objectives are not supported by XGBDDPM estimators."
            )

        X_np = np.ascontiguousarray(X, dtype=np.float32)
        y_np = np.ascontiguousarray(y, dtype=np.float32)
        self.alpha_bars = np.ascontiguousarray(self.alpha_bars, dtype=np.float32)
        params = self.get_xgb_params()
        model, metric, params, feature_weights = self._configure_fit(
            xgb_model, params, feature_weights
        )
        if feature_weights is not None:
            raise NotImplementedError("feature_weights are not supported by XGBDDPM estimators.")

        with config_context(verbosity=self.verbosity):
            dtrain = self._empty_dmatrix(X_np)
            self._refresh_dmatrix(dtrain, X_np, y_np, 0)
            eval_arrays = (
                []
                if eval_set is None
                else [
                    (
                        np.ascontiguousarray(a, dtype=np.float32),
                        np.ascontiguousarray(b, dtype=np.float32),
                    )
                    for a, b in eval_set
                ]
            )
            evals = []
            for i, (a, b) in enumerate(eval_arrays):
                dm = self._empty_dmatrix(a)
                self._refresh_dmatrix(dm, a, b, 0, validation=True)
                evals.append((dm, f"validation_{i}"))
            bst = Booster(params, [dtrain] + [dm for dm, _ in evals], model_file=model)

            callbacks = (
                [] if self.callbacks is None else copy.copy(list(self.callbacks))
            )
            if verbose:
                callbacks.append(
                    EvaluationMonitor(period=1 if verbose is True else int(verbose))
                )
            if self.early_stopping_rounds:
                early_stop = EarlyStopping(rounds=self.early_stopping_rounds)
                min_round = int(getattr(self, "min_boosting_round", 0))
                callbacks.append(
                    _MinBoostingRound(early_stop, min_round)
                    if min_round > 0
                    else early_stop
                )
            cb = CallbackContainer(callbacks, metric=metric)

            bst = cb.before_training(bst)
            refresh_every_k = int(self.refresh_every_k)
            for i in range(self.get_num_boosting_rounds()):
                if refresh_every_k >= 1 and i and i % refresh_every_k == 0:
                    self._refresh_dmatrix(dtrain, X_np, y_np, i)
                    for (dm, _), (a, b) in zip(evals, eval_arrays):
                        self._refresh_dmatrix(dm, a, b, i, validation=True)
                    _check_call(_LIB.XGBoosterClearCaches(bst.handle))
                if cb.before_iteration(bst, i, dtrain, evals):
                    break
                bst.update(dtrain, iteration=i)
                if cb.after_iteration(bst, i, dtrain, evals):
                    break
            bst = cb.after_training(bst)

        self._Booster = bst.reset()
        self._set_evaluation_result(cb.history)
        return self


class _MinBoostingRound(TrainingCallback):
    def __init__(self, callback: TrainingCallback, min_round: int) -> None:
        self.callback = callback
        self.min_round = min_round

    def before_training(self, model: Any) -> Any:
        return self.callback.before_training(model)

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        return self.callback.after_iteration(model, epoch, evals_log) and epoch + 1 >= self.min_round

    def after_training(self, model: Any) -> Any:
        return self.callback.after_training(model)


class XGBDDPMRegressor(_XGBDDPMMixin, XGBRegressor):
    def __init__(
        self,
        *,
        alpha_bars: Optional[Any] = None,
        n_classes: Optional[Any] = None,
        noise_samples_per_row: int = 1,
        timestep: Optional[int] = None,
        target_index: Optional[int] = None,
        ddpm_objective: str = "x",
        refresh_every_k: int = 1,
        min_boosting_round: int = 0,
        objective: Any = "reg:squarederror",
        **kwargs: Any,
    ) -> None:
        super().__init__(objective=objective, **kwargs)
        self.alpha_bars = alpha_bars
        self.n_classes = n_classes
        self.noise_samples_per_row = noise_samples_per_row
        self.timestep = timestep
        self.target_index = target_index
        self.ddpm_objective = str(ddpm_objective).lower()
        self.refresh_every_k = refresh_every_k
        self.min_boosting_round = min_boosting_round

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "XGBDDPMRegressor":
        return self._fit_ddpm(X, y, **kwargs)


class XGBDDPMClassifier(_XGBDDPMMixin, XGBClassifier):
    def __init__(
        self,
        *,
        alpha_bars: Optional[Any] = None,
        n_classes: Optional[Any] = None,
        noise_samples_per_row: int = 1,
        timestep: Optional[int] = None,
        target_index: Optional[int] = None,
        ddpm_objective: str = "x",
        refresh_every_k: int = 1,
        min_boosting_round: int = 0,
        objective: Any = "binary:logistic",
        **kwargs: Any,
    ) -> None:
        super().__init__(objective=objective, **kwargs)
        self.alpha_bars = alpha_bars
        self.n_classes = n_classes
        self.noise_samples_per_row = noise_samples_per_row
        self.timestep = timestep
        self.target_index = target_index
        self.ddpm_objective = str(ddpm_objective).lower()
        self.refresh_every_k = refresh_every_k
        self.min_boosting_round = min_boosting_round

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "XGBDDPMClassifier":
        classes = np.unique(np.asarray(y))
        self.n_classes_ = len(classes)
        expected = self.classes_
        if classes.shape != expected.shape or not (classes == expected).all():
            raise ValueError(
                "Invalid classes inferred from unique values of `y`. "
                f"Expected: {expected}, got {classes}"
            )
        if self.n_classes_ > 2 and self.objective != "multi:softmax":
            self.objective = "multi:softprob"
            self.kwargs = getattr(self, "kwargs", {})
            self.kwargs["num_class"] = self.n_classes_
        return self._fit_ddpm(X, y, **kwargs)
