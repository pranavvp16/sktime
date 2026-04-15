"""Adapter for using MOIRAI Forecasters."""

from unittest.mock import patch

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster
from sktime.utils.singleton import _multiton


@_multiton
class _CachedMoirai:
    """Cached MOIRAI model, ensuring one instance per unique configuration.

    MOIRAI is a zero-shot foundation model; loading it is expensive.
    The multiton ensures that multiple forecaster instances sharing the same
    checkpoint and hyperparameters reuse one loaded model rather than each
    loading their own copy.
    """

    def __init__(
        self,
        key,
        checkpoint_path,
        context_length,
        patch_size,
        num_samples,
        target_dim,
        feat_dynamic_real_dim,
        past_feat_dynamic_real_dim,
        map_location,
        use_source_package,
        prediction_length,
    ):
        self.key = key
        self.checkpoint_path = checkpoint_path
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.map_location = map_location
        self.use_source_package = use_source_package
        self.prediction_length = prediction_length
        self.model_ = None

    def load_model(self):
        """Load and return model_, reusing if already loaded."""
        if self.model_ is not None:
            return self.model_

        model_kwargs = {
            "prediction_length": self.prediction_length,
            "context_length": self.context_length,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "target_dim": self.target_dim,
            "feat_dynamic_real_dim": self.feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": self.past_feat_dynamic_real_dim,
        }

        if self.use_source_package:
            if _check_soft_dependencies("uni2ts", severity="none"):
                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

                if self.checkpoint_path.startswith("Salesforce"):
                    model_kwargs["module"] = MoiraiModule.from_pretrained(
                        self.checkpoint_path
                    )
                    self.model_ = MoiraiForecast(**model_kwargs)
                else:
                    from huggingface_hub import hf_hub_download

                    model_kwargs["checkpoint_path"] = hf_hub_download(
                        repo_id=self.checkpoint_path, filename="model.ckpt"
                    )
                    self.model_ = MoiraiForecast.load_from_checkpoint(
                        **model_kwargs, weights_only=False
                    )
        else:
            import sktime.libs.uni2ts as _uni2ts

            with patch.dict("sys.modules", {"uni2ts": _uni2ts}):
                from sktime.libs.uni2ts.forecast import MoiraiForecast

                if self.checkpoint_path.startswith("Salesforce"):
                    from sktime.libs.uni2ts.moirai_module import MoiraiModule

                    model_kwargs["module"] = MoiraiModule.from_pretrained(
                        self.checkpoint_path
                    )
                    self.model_ = MoiraiForecast(**model_kwargs)
                else:
                    from huggingface_hub import hf_hub_download

                    model_kwargs["checkpoint_path"] = hf_hub_download(
                        repo_id=self.checkpoint_path, filename="model.ckpt"
                    )
                    self.model_ = MoiraiForecast.load_from_checkpoint(
                        **model_kwargs, weights_only=False
                    )

        self.model_.to(self.map_location)
        return self.model_


__author__ = ["gorold", "chenghaoliu89", "liu-jc", "benheid", "pranavvp16"]
# gorold, chenghaoliu89, liu-jc are from SalesforceAIResearch/uni2ts


class MOIRAIForecaster(_BaseGlobalForecaster):
    """
    Adapter for using MOIRAI Forecasters.

    Parameters
    ----------
    checkpoint_path : str, default=None
        Path to the checkpoint of the model. Supported weights are available at [1]_.
    context_length : int, default=200
        Length of the context window: time points the model takes as input for
        inference.
    patch_size : int, default=32
        Time steps to perform patching with.
    num_samples : int, default=100
        Number of samples to draw.
    map_location : str, default=None
        Hardware to use for the model.
    target_dim : int, default=2
        Dimension of the target.
    deterministic : bool, default=False
        Whether to use a deterministic model.
    batch_size : int, default=32
        Number of samples in each batch of inference.
    broadcasting : bool, default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``
     use_source_package : bool, default=False
        If True, the model and configuration will be loaded directly from the source
        package ``uni2ts.models.moirai``. This is useful if you
        want to bypass the local version of the package or when working in an
        environment where the latest updates from the source package are needed.
        If False, the model and configuration will be loaded from the local
        version of package maintained in sktime.
        To install the source package, follow the instructions here [2]_.

    Examples
    --------
    >>> from sktime.forecasting.moirai_forecaster import MOIRAIForecaster
    >>> import pandas as pd
    >>> import numpy as np
    >>> morai_forecaster = MOIRAIForecaster(
    ...     checkpoint_path=f"sktime/moirai-1.0-R-small"
    ... )
    >>> y = np.random.normal(0, 1, (30, 2))
    >>> X = y * 2 + np.random.normal(0, 1, (30,1))
    >>> index = pd.date_range("2020-01-01", periods=30, freq="D")
    >>> y = pd.DataFrame(y, index=index)
    >>> X = pd.DataFrame(X, columns=["x1", "x2"], index=index)
    >>> morai_forecaster.fit(y, X=X)
    MOIRAIForecaster(checkpoint_path='sktime/moirai-1.0-R-small')
    >>> X_test = pd.DataFrame(np.random.normal(0, 1, (10, 2)),
    ...                      columns=["x1", "x2"],
    ...                      index=pd.date_range("2020-01-31", periods=10, freq="D"),
    ... )
    >>> forecast = morai_forecaster.predict(fh=range(1, 11), X=X_test)

    References
    ----------
    .. [1] https://huggingface.co/collections/sktime/moirai-variations-66ba3bc9f1dfeeafaed3b974
    .. [2] https://pypi.org/project/uni2ts/1.1.0/
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["gorold", "chenghaoliu89", "liu-jc", "benheid", "pranavvp16"],
        # gorold, chenghaoliu89, liu-jc are from SalesforceAIResearch/uni2ts
        "maintainers": ["pranavvp16"],
        "python_version": "<3.14",
        "python_dependencies": [
            "gluonts",
            "torch",
            "einops",
            "huggingface_hub",
            "hf-xet",
            "lightning",
            "hydra-core",
        ],
        # estimator type
        # --------------
        "scitype:y": "both",
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": True,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:pretrain": True,
        "capability:global_forecasting": True,
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        checkpoint_path: str,
        context_length=200,
        patch_size=32,
        num_samples=100,
        num_feat_dynamic_real=None,
        num_past_feat_dynamic_real=None,
        map_location=None,
        target_dim=2,
        broadcasting=False,
        deterministic=False,
        batch_size=32,
        use_source_package=False,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real
        self.map_location = map_location
        self.target_dim = target_dim
        self.broadcasting = broadcasting
        self.deterministic = deterministic
        self.batch_size = batch_size
        self.use_source_package = use_source_package

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.DataFrame",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

    # Apply a patch for redirecting imports to sktime.libs.uni2ts
    if _check_soft_dependencies(["lightning", "huggingface-hub"], severity="none"):
        import sktime
        from sktime.libs.uni2ts.forecast import MoiraiForecast

        @patch.dict("sys.modules", {"uni2ts": sktime.libs.uni2ts})
        def _instantiate_patched_model(self, model_kwargs):
            """Instantiate the model from the vendor package."""
            from sktime.libs.uni2ts.forecast import MoiraiForecast

            if self.checkpoint_path.startswith("Salesforce"):
                from sktime.libs.uni2ts.moirai_module import MoiraiModule

                model_kwargs["module"] = MoiraiModule.from_pretrained(
                    self.checkpoint_path
                )
                return MoiraiForecast(**model_kwargs)
            else:
                from huggingface_hub import hf_hub_download

                model_kwargs["checkpoint_path"] = hf_hub_download(
                    repo_id=self.checkpoint_path, filename="model.ckpt"
                )
                return MoiraiForecast.load_from_checkpoint(
                    **model_kwargs, weights_only=False
                )

    def _get_moirai_cache_key(
        self,
        prediction_length,
        feat_dynamic_real_dim,
        past_feat_dynamic_real_dim,
        target_dim=None,
    ):
        """Return a hashable key identifying this model configuration."""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "context_length": self.context_length,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "target_dim": target_dim if target_dim is not None else self.target_dim,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "map_location": str(self.map_location),
            "use_source_package": self.use_source_package,
            "prediction_length": prediction_length,
        }
        return str(sorted(config.items()))

    def _load_model_via_multiton(
        self,
        prediction_length,
        feat_dynamic_real_dim,
        past_feat_dynamic_real_dim,
        target_dim=None,
    ):
        """Load or retrieve model from the _CachedMoirai multiton."""
        if target_dim is None:
            target_dim = self.target_dim
        cache_key = self._get_moirai_cache_key(
            prediction_length,
            feat_dynamic_real_dim,
            past_feat_dynamic_real_dim,
            target_dim,
        )
        self.model_ = _CachedMoirai(
            key=cache_key,
            checkpoint_path=self.checkpoint_path,
            context_length=self.context_length,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            map_location=self.map_location,
            use_source_package=self.use_source_package,
            prediction_length=prediction_length,
        ).load_model()

    def _pretrain(self, y, X=None, fh=None):
        """Load the MOIRAI foundation model from checkpoint (pretrain step).

        This implements the first stage of the 3-state sktime pretrain API:
        "new" → "pretrained". The model is loaded once here and reused in
        subsequent ``fit()`` calls without reloading.

        MOIRAI is a zero-shot foundation model; this method performs model
        loading (the expensive step), not fine-tuning.

        Parameters
        ----------
        y : pd.DataFrame
            Panel or hierarchical time series data. Not used for loading but
            required by the pretrain API signature.
        X : pd.DataFrame, optional (default=None)
            Exogenous data. Used only to infer ``feat_dynamic_real_dim`` when
            ``num_feat_dynamic_real`` is not set explicitly.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon used to set ``prediction_length``.

        Returns
        -------
        self : reference to self
        """
        if fh is not None and self._cutoff is not None:
            prediction_length = int(max(fh.to_relative(self._cutoff)))
        else:
            prediction_length = int(getattr(self, "_pretrain_prediction_length_", 1))
        self._pretrain_prediction_length_ = prediction_length

        feat_dynamic_real_dim = (
            self.num_feat_dynamic_real
            if self.num_feat_dynamic_real is not None
            else (X.shape[1] if X is not None else 0)
        )
        past_feat_dynamic_real_dim = (
            self.num_past_feat_dynamic_real
            if self.num_past_feat_dynamic_real is not None
            else 0
        )

        # For multivariate data (>1 col), override target_dim with actual column count.
        # For univariate (Series or 1-col DataFrame), keep self.target_dim as-is.
        effective_target_dim = (
            y.shape[1]
            if isinstance(y, pd.DataFrame) and y.shape[1] > 1
            else self.target_dim
        )

        self._load_model_via_multiton(
            prediction_length,
            feat_dynamic_real_dim,
            past_feat_dynamic_real_dim,
            effective_target_dim,
        )
        return self

    def _pretrain_update(self, y, X=None, fh=None):
        """Re-load the model with updated parameters (pretrain update step).

        Parameters
        ----------
        y : pd.DataFrame
            Panel or hierarchical time series data.
        X : pd.DataFrame, optional (default=None)
            Exogenous data.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        return self._pretrain(y=y, X=X, fh=fh)

    def _fit(self, y, X, fh):
        if fh is not None:
            prediction_length = int(max(fh.to_relative(self.cutoff)))
        else:
            prediction_length = int(getattr(self, "_pretrain_prediction_length_", 1))

        feat_dynamic_real_dim = (
            self.num_feat_dynamic_real
            if self.num_feat_dynamic_real is not None
            else (X.shape[1] if X is not None else 0)
        )
        past_feat_dynamic_real_dim = (
            self.num_past_feat_dynamic_real
            if self.num_past_feat_dynamic_real is not None
            else 0
        )

        # For multivariate data (>1 col), override target_dim with actual column count.
        effective_target_dim = (
            y.shape[1]
            if isinstance(y, pd.DataFrame) and y.shape[1] > 1
            else self.target_dim
        )

        # If pretrained, reuse the already-loaded model — skip expensive reload
        if hasattr(self, "model_"):
            self.model_.hparams["prediction_length"] = prediction_length
            return self

        # Zero-shot path (no prior pretrain): load via multiton
        self._load_model_via_multiton(
            prediction_length,
            feat_dynamic_real_dim,
            past_feat_dynamic_real_dim,
            effective_target_dim,
        )

    def _predict(self, fh, y=None, X=None):
        if self.deterministic:
            import torch

            try:
                torch.manual_seed(42)
            except RuntimeError as e:
                if "TORCH_LIBRARY" not in str(e):
                    raise

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        if min(fh._values) < 0:
            raise NotImplementedError(
                "The MORAI adapter is not supporting insample predictions."
            )

        _y = self._y.copy()
        _X = None
        if self._X is not None:
            _X = self._X.copy()

        # Zero shot case with X and fit data as context
        _use_fit_data_as_context = False
        if X is not None and y is None:
            _use_fit_data_as_context = True

        # Override to data in fit as new timeseries is passed
        elif y is not None:
            _y = y.copy()
            if X is not None:
                _X = X.copy()

        if isinstance(_y, pd.Series):
            target = [_y.name]
            _y, _is_converted_to_df = self._series_to_df(_y)
        else:
            target = _y.columns

        # Store the original index and target name
        self._target_name = target
        self._len_of_targets = len(target)

        target = [f"target_{i}" for i in range(self._len_of_targets)]
        _y.columns = target

        future_length = 0
        feat_dynamic_real = None

        if _X is not None:
            feat_dynamic_real = [
                f"feat_dynamic_real_{i}" for i in range(self._X.shape[1])
            ]
            _X.columns = feat_dynamic_real

        pred_df = pd.concat([_y, _X], axis=1)
        self._is_range_index = self.check_range_index(pred_df)
        self._is_period_index = self.check_period_index(pred_df)

        if _use_fit_data_as_context:
            future_length = self._get_future_length(X)
            first_seen_index = X.index[0]
            X_to_extend = X.copy()
            X_to_extend.columns = feat_dynamic_real
            _X = pd.concat([_X, X_to_extend]).sort_index()
            pred_df = pd.concat([_y, _X], axis=1).sort_index()
            pred_df.fillna(0, inplace=True)

            # Ensure time index is continuous — non-contiguous fh (e.g. [2, 5])
            # produces gaps in X_test; reindex to fill so GluonTS can infer freq.
            if not isinstance(pred_df.index, pd.MultiIndex):
                time_index = pred_df.index
                if len(time_index) > 1 and isinstance(
                    time_index, (pd.DatetimeIndex, pd.PeriodIndex)
                ):
                    freq = getattr(time_index, "freq", None)
                    if freq is None and len(time_index) >= 3:
                        freq = pd.infer_freq(time_index[:3])
                    if freq is not None:
                        if isinstance(time_index, pd.PeriodIndex):
                            continuous_index = pd.period_range(
                                start=time_index.min(),
                                end=time_index.max(),
                                freq=freq,
                            )
                        else:
                            continuous_index = pd.date_range(
                                start=time_index.min(),
                                end=time_index.max(),
                                freq=freq,
                            )
                        if len(continuous_index) != len(time_index):
                            pred_df = pred_df.reindex(continuous_index)
                            if feat_dynamic_real is not None:
                                pred_df[feat_dynamic_real] = pred_df[
                                    feat_dynamic_real
                                ].ffill()
                            pred_df.fillna(0, inplace=True)
                            future_length = len(pred_df) - len(_y)
        else:
            if _X is not None:
                future_length = len(_X.index.get_level_values(-1).unique()) - len(
                    _y.index.get_level_values(-1).unique()
                )
            else:
                future_length = 0
        # check whether the index is a PeriodIndex
        if isinstance(pred_df.index, pd.PeriodIndex):
            time_idx = self.return_time_index(pred_df)
            pred_df.index = time_idx.to_timestamp()
            pred_df.index.freq = None

        # Check if the index is a range index
        if self._is_range_index:
            pred_df.index = self.handle_range_index(pred_df.index)

        _is_hierarchical = False
        if pred_df.index.nlevels >= 3:
            pred_df = self._convert_hierarchical_to_panel(pred_df)
            _is_hierarchical = True

        ds_test, df_config = self.create_pandas_dataset(
            pred_df, target, feat_dynamic_real, future_length
        )

        with self.model_.hparams_context(prediction_length=int(max(fh._values))):
            predictor = self.model_.create_predictor(batch_size=self.batch_size)
            forecasts = list(predictor.predict(ds_test))
        forecast_it = iter(forecasts)
        predictions = self._get_prediction_df(forecast_it, df_config)
        if isinstance(_y.index.get_level_values(-1), pd.DatetimeIndex):
            if isinstance(predictions.index, pd.MultiIndex):
                predictions.index = predictions.index.set_levels(
                    levels=predictions.index.get_level_values(-1)
                    .to_timestamp()
                    .unique(),
                    level=-1,
                )
            else:
                predictions.index = predictions.index.to_timestamp()
        if _is_hierarchical:
            predictions = self._convert_panel_to_hierarchical(
                predictions, _y.index.names
            )

        pred_out = fh.get_expected_pred_idx(_y, cutoff=self.cutoff)

        if self._is_range_index:
            timepoints = self.return_time_index(predictions)
            timepoints = timepoints.to_timestamp()
            timepoints = (timepoints - pd.Timestamp("2010-01-01")).map(
                lambda x: x.days
            ) + self.return_time_index(_y)[0]
            if isinstance(predictions.index, pd.MultiIndex):
                predictions.index = predictions.index.set_levels(
                    levels=timepoints.unique(), level=-1
                )
                # Convert str type to int
                predictions.index = predictions.index.map(lambda x: (int(x[0]), x[1]))
            else:
                predictions.index = timepoints

        if _use_fit_data_as_context:
            predictions = predictions.loc[first_seen_index:]

        predictions = predictions.loc[pred_out]
        predictions.index = pred_out
        return predictions

    def _predict_quantiles(self, fh, X=None, alpha=None, y=None):
        """Compute quantile forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series.
        alpha : list of float, optional (default=None)
            The quantiles to predict. If None, uses default [0.1, 0.25, 0.5, 0.75, 0.9].
        y : pd.DataFrame, optional (default=None)
            Historical values of the time series for global forecasting.

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts with MultiIndex columns (variable, alpha).
        """
        if alpha is None:
            alpha = [0.1, 0.25, 0.5, 0.75, 0.9]

        if self.deterministic:
            try:
                import torch

                torch.manual_seed(42)
            except RuntimeError as e:
                if "TORCH_LIBRARY" not in str(e):
                    raise

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        if min(fh._values) < 0:
            raise NotImplementedError(
                "The MORAI adapter is not supporting insample predictions."
            )

        _y = self._y.copy()
        _X = None
        if self._X is not None:
            _X = self._X.copy()

        _use_fit_data_as_context = False
        if X is not None and y is None:
            _use_fit_data_as_context = True
        elif y is not None:
            _y = y.copy()
            if X is not None:
                _X = X.copy()

        if isinstance(_y, pd.Series):
            target = [_y.name]
            _y, _is_converted_to_df = self._series_to_df(_y)
        else:
            target = _y.columns

        self._target_name = target
        self._len_of_targets = len(target)

        target = [f"target_{i}" for i in range(self._len_of_targets)]
        _y.columns = target

        future_length = 0
        feat_dynamic_real = None

        if _X is not None:
            feat_dynamic_real = [
                f"feat_dynamic_real_{i}" for i in range(self._X.shape[1])
            ]
            _X.columns = feat_dynamic_real

        pred_df = pd.concat([_y, _X], axis=1)

        if _X is not None and not _use_fit_data_as_context:
            time_index = self.return_time_index(pred_df)
            if len(time_index) > 1:
                if isinstance(time_index, (pd.DatetimeIndex, pd.PeriodIndex)):
                    freq = time_index.freq
                    if freq is None:
                        freq = (
                            pd.infer_freq(time_index[:3])
                            if len(time_index) >= 3
                            else None
                        )

                    if freq is not None:
                        if isinstance(time_index, pd.PeriodIndex):
                            continuous_index = pd.period_range(
                                start=time_index.min(), end=time_index.max(), freq=freq
                            )
                        else:
                            continuous_index = pd.date_range(
                                start=time_index.min(), end=time_index.max(), freq=freq
                            )

                        if not isinstance(pred_df.index, pd.MultiIndex):
                            pred_df = pred_df.reindex(continuous_index)
                            if feat_dynamic_real is not None:
                                pred_df[feat_dynamic_real] = pred_df[
                                    feat_dynamic_real
                                ].ffill()

        self._is_range_index = self.check_range_index(pred_df)
        self._is_period_index = self.check_period_index(pred_df)

        if _use_fit_data_as_context:
            future_length = self._get_future_length(X)
            first_seen_index = X.index[0]
            X_to_extend = X.copy()
            if feat_dynamic_real is None:
                feat_dynamic_real = [
                    f"feat_dynamic_real_{i}" for i in range(X.shape[1])
                ]
            X_to_extend.columns = feat_dynamic_real
            if _X is not None:
                _X = pd.concat([_X, X_to_extend]).sort_index()
            else:
                _X = X_to_extend
            pred_df = pd.concat([_y, _X], axis=1).sort_index()
            pred_df.fillna(0, inplace=True)

            if not isinstance(pred_df.index, pd.MultiIndex):
                time_index = pred_df.index
                if len(time_index) > 1 and isinstance(
                    time_index, (pd.DatetimeIndex, pd.PeriodIndex)
                ):
                    freq = time_index.freq
                    if freq is None and len(time_index) >= 2:
                        if isinstance(time_index, pd.DatetimeIndex):
                            freq = pd.infer_freq(time_index[: min(3, len(time_index))])
                        elif isinstance(time_index, pd.PeriodIndex):
                            freq = time_index.freq

                    if freq is not None:
                        if isinstance(time_index, pd.PeriodIndex):
                            continuous_index = pd.period_range(
                                start=time_index.min(), end=time_index.max(), freq=freq
                            )
                        else:
                            continuous_index = pd.date_range(
                                start=time_index.min(), end=time_index.max(), freq=freq
                            )

                        if len(continuous_index) != len(time_index):
                            pred_df = pred_df.reindex(continuous_index)
                            if feat_dynamic_real is not None:
                                pred_df[feat_dynamic_real] = pred_df[
                                    feat_dynamic_real
                                ].ffill()
                            pred_df.fillna(0, inplace=True)
        else:
            if _X is not None:
                future_length = len(_X.index.get_level_values(-1).unique()) - len(
                    _y.index.get_level_values(-1).unique()
                )
            else:
                future_length = 0

        _original_period_freq = None
        if isinstance(pred_df.index, pd.PeriodIndex):
            time_idx = self.return_time_index(pred_df)
            _original_period_freq = time_idx.freq
            pred_df.index = time_idx.to_timestamp()
            pred_df.index.freq = None

        if self._is_range_index:
            pred_df.index = self.handle_range_index(pred_df.index)

        _is_hierarchical = False
        if pred_df.index.nlevels >= 3:
            pred_df = self._convert_hierarchical_to_panel(pred_df)
            _is_hierarchical = True

        ds_test, df_config = self.create_pandas_dataset(
            pred_df, target, feat_dynamic_real, future_length
        )

        with self.model_.hparams_context(prediction_length=int(max(fh._values))):
            predictor = self.model_.create_predictor(batch_size=self.batch_size)
            forecast_list = list(predictor.predict(ds_test))

        quantile_dfs = []

        for forecast in forecast_list:
            # Use forecast.index directly to avoid mean_ts crashing on 2D arrays
            time_index = forecast.index if hasattr(forecast, "index") else None

            for q in alpha:
                try:
                    q_result = forecast.quantile(q)

                    if isinstance(q_result, np.ndarray) and q_result.ndim == 2:
                        # Multivariate: shape (prediction_length, n_targets)
                        for col_idx in range(q_result.shape[1]):
                            col_name = (
                                self._target_name[col_idx]
                                if self._target_name is not None
                                and col_idx < len(self._target_name)
                                else col_idx
                            )
                            if col_name is None:
                                col_name = col_idx
                            q_col = pd.Series(q_result[:, col_idx], index=time_index)
                            if forecast.item_id is not None:
                                df = q_col.reset_index()
                                df.columns = [df_config["timepoints"], "quantile"]
                                df["alpha"] = q
                                df["__variable__"] = col_name
                                df[df_config["item_id"]] = forecast.item_id
                                quantile_dfs.append(df)
                            else:
                                df = q_col.to_frame(name="quantile")
                                df["alpha"] = q
                                df["__variable__"] = col_name
                                df = df.reset_index()
                                df.columns = [
                                    "timepoints",
                                    "quantile",
                                    "alpha",
                                    "__variable__",
                                ]
                                quantile_dfs.append(df)
                    else:
                        # Univariate
                        col_name = (
                            self._target_name[0]
                            if self._target_name is not None
                            and len(self._target_name) > 0
                            else 0
                        )
                        if col_name is None:
                            col_name = 0

                        if isinstance(q_result, pd.Series):
                            q_series = q_result
                        elif time_index is not None:
                            q_series = pd.Series(q_result, index=time_index)
                        else:
                            pred_length = (
                                len(q_result) if hasattr(q_result, "__len__") else 1
                            )
                            q_series = pd.Series(q_result, index=range(pred_length))

                        if forecast.item_id is not None:
                            df = q_series.reset_index()
                            df.columns = [df_config["timepoints"], "quantile"]
                            df["alpha"] = q
                            df["__variable__"] = col_name
                            df[df_config["item_id"]] = forecast.item_id
                            quantile_dfs.append(df)
                        else:
                            df = q_series.to_frame(name="quantile")
                            df["alpha"] = q
                            df["__variable__"] = col_name
                            df = df.reset_index()
                            df.columns = [
                                "timepoints",
                                "quantile",
                                "alpha",
                                "__variable__",
                            ]
                            quantile_dfs.append(df)
                except (AttributeError, TypeError):
                    continue

        if len(quantile_dfs) > 0:
            result = pd.concat(quantile_dfs, ignore_index=True)

            if forecast_list[0].item_id is not None:
                result = result.pivot_table(
                    values="quantile",
                    index=[df_config["item_id"], df_config["timepoints"]],
                    columns=["__variable__", "alpha"],
                )
                result.columns.names = ["variable", "alpha"]
            else:
                result = result.pivot_table(
                    values="quantile",
                    index="timepoints",
                    columns=["__variable__", "alpha"],
                )
                result.columns.names = ["variable", "alpha"]

            if isinstance(_y.index.get_level_values(-1), pd.DatetimeIndex):
                if isinstance(result.index, pd.MultiIndex):
                    result.index = result.index.set_levels(
                        levels=result.index.get_level_values(-1)
                        .to_timestamp()
                        .unique(),
                        level=-1,
                    )
                else:
                    result.index = result.index.to_timestamp()

            if _is_hierarchical:
                original_columns = result.columns
                # Temporarily flatten MultiIndex columns so reset_index() inside
                # _convert_panel_to_hierarchical doesn't create tuple column names
                # that prevent set_index from finding the time level by name.
                flat_names = [f"__col_{i}__" for i in range(len(result.columns))]
                result.columns = flat_names
                result = self._convert_panel_to_hierarchical(result, _y.index.names)
                result.columns = original_columns

            pred_out = fh.get_expected_pred_idx(_y, cutoff=self.cutoff)

            if self._is_range_index:
                timepoints = self.return_time_index(result)
                timepoints = timepoints.to_timestamp()
                timepoints = (timepoints - pd.Timestamp("2010-01-01")).map(
                    lambda x: x.days
                ) + self.return_time_index(_y)[0]
                if isinstance(result.index, pd.MultiIndex):
                    result.index = result.index.set_levels(
                        levels=timepoints.unique(), level=-1
                    )
                    result.index = result.index.map(lambda x: (int(x[0]), x[1]))
                else:
                    result.index = timepoints

            if _use_fit_data_as_context:
                result = result.loc[first_seen_index:]

            pred_out_for_loc = pred_out
            if isinstance(result.index, pd.DatetimeIndex) and isinstance(
                pred_out, pd.PeriodIndex
            ):
                pred_out_for_loc = pred_out.to_timestamp()
            elif isinstance(result.index, pd.PeriodIndex) and isinstance(
                pred_out, pd.DatetimeIndex
            ):
                pred_out_for_loc = pred_out.to_period(freq=result.index.freq)

            try:
                result = result.loc[pred_out_for_loc]
            except (KeyError, IndexError):
                result = result.reindex(pred_out_for_loc)

            result.index = pred_out
            return result

        return pd.DataFrame()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {
                "deterministic": True,
                "checkpoint_path": "sktime/moirai-1.0-R-small",
            },
            {
                "deterministic": True,
                "checkpoint_path": "sktime/moirai-1.0-R-small",
                "num_samples": 50,
                "context_length": 100,
            },
        ]

    def _get_prediction_df(self, forecast_iter, df_config):
        def handle_series_prediction(forecast, target):
            # forecast.mean has shape (pred_len,) for univariate or
            # (pred_len, target_dim) for multivariate. Use it to avoid
            # the pd.Series constructor error on 2D arrays in mean_ts.
            mean_vals = forecast.mean
            if mean_vals.ndim == 2:
                # Multivariate: build DataFrame with original target column names
                pred = pd.DataFrame(
                    mean_vals,
                    index=forecast.index,
                    columns=[t if t is not None else i for i, t in enumerate(target)],
                )
                return pred
            else:
                # Univariate
                pred = forecast.mean_ts
                if target[0] is not None:
                    return pred.rename(target[0])
                else:
                    return pred

        def handle_panel_predictions(forecasts_it, df_config):
            # Convert all panel forecasts to a single panel dataframe
            panels = []
            for forecast in forecasts_it:
                mean_vals = forecast.mean
                if mean_vals.ndim == 2:
                    # Multivariate: mean has shape (prediction_length, n_targets)
                    df = pd.DataFrame(
                        mean_vals,
                        index=forecast.index,
                        columns=list(df_config["target"]),
                    )
                    df = df.reset_index()
                    df.columns = [df_config["timepoints"]] + list(df_config["target"])
                else:
                    # Univariate
                    df = forecast.mean_ts.reset_index()
                    df.columns = [df_config["timepoints"], df_config["target"][0]]
                df[df_config["item_id"]] = forecast.item_id
                df.set_index(
                    [df_config["item_id"], df_config["timepoints"]], inplace=True
                )
                panels.append(df)
            return pd.concat(panels)

        forecasts = list(forecast_iter)

        # Assuming all forecasts_it are either series or panel type.
        if forecasts[0].item_id is None:
            return handle_series_prediction(forecasts[0], df_config["target"])
        else:
            return handle_panel_predictions(forecasts, df_config)

    def create_pandas_dataset(
        self, df, target, dynamic_features=None, forecast_horizon=0
    ):
        """Create a gluonts PandasDataset from the input data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        target : str
            Target column name.
        dynamic_features : list, default=None
            List of dynamic features.
        forecast_horizon : int, default=0
            Forecast horizon.

        Returns
        -------
        dataset : PandasDataset
            Pandas dataset.
        df_config : dict
            Configuration of the input data.

        """
        if _check_soft_dependencies("gluonts", severity="none"):
            from gluonts.dataset.pandas import PandasDataset

        # Add original target to config
        df_config = {
            "target": self._target_name,
        }

        # PandasDataset expects non-multiindex dataframe with item_id
        # and timepoints
        if isinstance(df.index, pd.MultiIndex):
            if None in df.index.names:
                df.index.names = ["item_id", "timepoints"]
            item_id = df.index.names[0]
            df_config["item_id"] = item_id
            timepoints = df.index.names[-1]
            df_config["timepoints"] = timepoints

            # Reset index to create a non-multiindex dataframe
            df = df.reset_index()
            df.set_index(timepoints, inplace=True)

            dataset = PandasDataset.from_long_dataframe(
                df,
                target=target,
                feat_dynamic_real=dynamic_features,
                item_id=item_id,
                future_length=forecast_horizon,
            )
        else:
            dataset = PandasDataset(
                df,
                target=target,
                feat_dynamic_real=dynamic_features,
                future_length=forecast_horizon,
            )

        return dataset, df_config

    # def _extend_df(self, df, _y, X=None):
    #     """Extend the input dataframe up to the timepoints that need to be predicted.
    #
    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Input data that needs to be extended
    #     X : pd.DataFrame, default=None
    #         Assumes that X has future timepoints and is
    #         concatenated to the input data,
    #         if X is present in the input, but None here the values of X are assumed
    #         to be 0 in future timepoints that need to be predicted.
    #     is_range_index : bool, default=False
    #         If True, the index is a range index.
    #     is_period_index : bool, default=False
    #         If True, the index is a period index.
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #         Extended dataframe with future timepoints.
    #     """
    #     index = self.return_time_index(df)
    #
    #     # Extend the index to the future timepoints
    #     # respective to index last seen
    #
    #     if self._is_range_index:
    #         pred_index = pd.RangeIndex(
    #             self.cutoff[0] + 1, self.cutoff[0] + max(self.fh._values)
    #         )
    #     elif self._is_period_index:
    #         pred_index = pd.period_range(
    #             self.cutoff[0],
    #             periods=max(self.fh._values) + 1,
    #             freq=index.freq,
    #         )[1:]
    #     else:
    #         pred_index = pd.date_range(
    #             self.cutoff[0],
    #             periods=max(self.fh._values) + 1,
    #             freq=self.infer_freq(index),
    #         )[1:]
    #
    #     if isinstance(df.index, pd.MultiIndex):
    #         # Works for any number of levels in the MultiIndex
    #         index_levels = [
    #             df.index.get_level_values(i).unique()
    #             for i in range(df.index.nlevels - 1)
    #         ]
    #         index_levels.append(pred_index)
    #         new_index = pd.MultiIndex.from_product(index_levels, names=df.index.names)
    #     else:
    #         new_index = pred_index
    #
    #     df_y = pd.DataFrame(columns=_y.columns, index=new_index)
    #     df_y.fillna(0, inplace=True)
    #     pred_df = pd.concat([df_y, X], axis=1)
    #     extended_df = pd.concat([df, pred_df])
    #     extended_df = extended_df.sort_index()
    #     extended_df.fillna(0, inplace=True)
    #
    #     return extended_df, df_y

    def infer_freq(self, index):
        """
        Infer frequency of the index.

        Parameters
        ----------
        index: pd.Index
            Index of the time series data.

        Notes
        -----
        Uses only first 3 values of the index to infer the frequency.
        As `freq=None` is returned in case of multiindex timepoints.

        """
        if isinstance(index, pd.PeriodIndex):
            return index.freq
        return pd.infer_freq(index[:3])

    def return_time_index(self, df):
        """Return the time index, given any type of index."""
        if isinstance(df.index, pd.MultiIndex):
            return df.index.get_level_values(-1)
        else:
            return df.index

    def check_range_index(self, df):
        """Check if the index is a range index."""
        timepoints = self.return_time_index(df)
        if isinstance(timepoints, pd.RangeIndex):
            return True
        elif pd.api.types.is_integer_dtype(timepoints):
            return True
        return False

    def check_period_index(self, df):
        """Check if the index is a PeriodIndex."""
        timepoints = self.return_time_index(df)
        if isinstance(timepoints, pd.PeriodIndex):
            return True
        return False

    def handle_range_index(self, index):
        """
        Convert RangeIndex to Dummy DatetimeIndex.

        As gluonts PandasDataset expects a DatetimeIndex.
        """
        start_date = "2010-01-01"
        if isinstance(index, pd.MultiIndex):
            n_periods = index.levels[-1].size
            datetime_index = pd.date_range(
                start=start_date, periods=n_periods, freq="D"
            )
            new_index = index.set_levels(datetime_index, level=-1)
        else:
            n_periods = index.size
            new_index = pd.date_range(start=start_date, periods=n_periods, freq="D")
        return new_index

    def _series_to_df(self, y):
        """Convert series to DataFrame."""
        is_converted = False
        if isinstance(y, pd.Series):
            y = y.to_frame()
            is_converted = True
        return y, is_converted

    def _convert_hierarchical_to_panel(self, df):
        # Flatten the MultiIndex to a panel type DataFrame
        data = df.copy()
        flattened_index = [("*".join(map(str, x[:-1])), x[-1]) for x in data.index]
        # Create a new MultiIndex with the flattened level and the last level unchanged
        data.index = pd.MultiIndex.from_tuples(
            flattened_index, names=["Flattened_Level", data.index.names[-1]]
        )
        return data

    def _convert_panel_to_hierarchical(self, df, original_index_names=None):
        # Store the original index names
        if original_index_names is None:
            original_index_names = df.index.names

        # Reset the index to get 'Flattened_Level' as a column
        data = df.reset_index()

        # Split the 'Flattened_Level' column into multiple columns
        split_levels = data["Flattened_Level"].str.split("*", expand=True)
        split_levels.columns = original_index_names[:-1]
        # Get the names of the split levels as a list of column names
        index_names = split_levels.columns.tolist()

        # Combine the split levels with the rest of the data
        data_converted = pd.concat(
            [split_levels, data.drop(columns=["Flattened_Level"])], axis=1
        )

        # Get the last index name if it exists, otherwise use a default name
        last_index_name = (
            original_index_names[-1]
            if original_index_names[-1] is not None
            else "timepoints"
        )

        # Set the new index with the split levels and the last index name
        data_converted = data_converted.set_index(index_names + [last_index_name])

        return data_converted

    def _get_future_length(self, X):
        """Get the future length."""
        if isinstance(X.index, pd.MultiIndex):
            return len(X.index.get_level_values(-1).unique())
        else:
            return len(X)
