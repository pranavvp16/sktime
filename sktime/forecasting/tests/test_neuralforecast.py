# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for interfacing estimators from neuralforecast."""
import pandas
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.neuralforecast import (
    NeuralForecastLSTM,
    NeuralForecastRNN,
    NeuralForecastAutoLSTM,
)
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class

__author__ = ["yarnabrina", "pranavvp16"]

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)


@pytest.mark.parametrize(
    "model_class", [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
)
@pytest.mark.skipif(
    not run_test_for_class(
        [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_univariate_y_without_X(model_class) -> None:
    """Test with single endogenous without exogenous."""
    # get params
    params = model_class.get_test_params()[0]

    # define model
    model = model_class(**params)

    # attempt fit with negative fh
    with pytest.raises(
        NotImplementedError, match="in-sample prediction is currently not supported"
    ):
        model.fit(y_train, fh=[-2, -1, 0, 1, 2])

    # train model
    model.fit(y_train, fh=[1, 2, 3, 4])

    # predict with trained model
    y_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(y_pred.index, y_test.index, check_names=False)


@pytest.mark.parametrize("model_class", [NeuralForecastLSTM, NeuralForecastRNN])
@pytest.mark.skipif(
    not run_test_for_class([NeuralForecastLSTM, NeuralForecastRNN]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_univariate_y_with_X(model_class) -> None:
    """Test with single endogenous with exogenous."""
    # get params
    params = model_class.get_test_params()[0]
    params["freq"] = "A-DEC"

    # select feature columns
    exog_list = ["GNP", "GNPDEFL", "UNEMP"]

    # define model
    model = model_class(**params, futr_exog_list=exog_list)

    # attempt fit without X
    with pytest.raises(
        ValueError, match="Missing exogeneous data, 'futr_exog_list' is non-empty."
    ):
        model.fit(y_train, fh=[1, 2, 3, 4])

    # train model with all X columns
    model.fit(y_train, X=X_train, fh=[1, 2, 3, 4])

    # attempt predict without X
    with pytest.raises(
        ValueError, match="Missing exogeneous data, 'futr_exog_list' is non-empty."
    ):
        model.predict()

    # predict with only selected columns
    # checking that rest are not used
    y_pred = model.predict(X=X_test[exog_list])

    # check prediction index
    pandas.testing.assert_index_equal(y_pred.index, y_test.index, check_names=False)


@pytest.mark.parametrize(
    "model_class", [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
)
@pytest.mark.skipif(
    not run_test_for_class(
        [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_multivariate_y_without_X(model_class) -> None:
    """Test with multiple endogenous without exogenous."""
    # get params
    params = model_class.get_test_params()[0]

    # define model
    model = model_class(**params)

    # train model
    model.fit(X_train, fh=[1, 2, 3, 4])

    # predict with trained model
    X_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(X_pred.index, X_test.index, check_names=False)


@pytest.mark.parametrize(
    "model_class", [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
)
@pytest.mark.skipif(
    not run_test_for_class(
        [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_with_non_default_loss(model_class) -> None:
    """Test with multiple endogenous without exogenous."""
    # get params
    params = model_class.get_test_params()[1]

    # define model
    model = model_class(**params)

    # train model
    model.fit(X_train, fh=[1, 2, 3, 4])

    # predict with trained model
    X_pred = model.predict()

    # check prediction index
    pandas.testing.assert_index_equal(X_pred.index, X_test.index, check_names=False)


@pytest.mark.parametrize(
    "model_class", [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
)
@pytest.mark.skipif(
    not run_test_for_class(
        [NeuralForecastLSTM, NeuralForecastRNN, NeuralForecastAutoLSTM]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_neural_forecast_fail_with_multiple_predictions(model_class) -> None:
    """Check fail when multiple prediction columns are used."""
    # import pytorch losses with multiple predictions capability
    from neuralforecast.losses.pytorch import MQLoss

    # get params
    params = model_class.get_test_params()[0]

    # define model
    model = model_class(**params, loss=MQLoss(quantiles=[0.25, 0.5, 0.75]))

    # train model
    model.fit(X_train, fh=[1, 2, 3, 4])

    # attempt predict
    with pytest.raises(
        NotImplementedError, match="Multiple prediction columns are not supported."
    ):
        model.predict()
