import numpy as np
import pandas as pd
import pytest

from src.features.calendar import FEATURES, add_calendar
from src.models.forecaster import forecast_errors, load, predict, save, train
from src.monitoring.drift import drift_scores, naive_reference, seasonal_reference


def test_forecast_errors():
    result = forecast_errors(np.array([10, 20, 0]), np.array([12, 18, 0]))
    assert result["mae"] == pytest.approx(4 / 3)
    assert result["bias_pct"] == pytest.approx(0.0)
    assert result["smape"] == pytest.approx(100 * (4 / 22 + 4 / 38 + 0) / 3)


def test_model_round_trips_through_a_file(sales, small_models, tmp_path):
    frame = add_calendar(sales)
    model = train(frame[frame["date"] < "2015-01-01"], FEATURES, n_jobs=1)
    save(model, tmp_path / "v001.ubj")
    reloaded = load(tmp_path / "v001.ubj")
    sample = frame.tail(50)
    np.testing.assert_allclose(predict(model, sample, FEATURES),
                               predict(reloaded, sample, FEATURES), rtol=1e-6)


@pytest.fixture
def distributions():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"sales": rng.gamma(4, 10, 3000), "prediction": rng.gamma(4, 10, 3000)})


def test_drift_flags_a_shift_and_ignores_noise(distributions):
    rng = np.random.default_rng(1)
    shifted = pd.DataFrame({"sales": rng.gamma(4, 10, 900) * 1.3,
                            "prediction": rng.gamma(4, 10, 900)})
    scores = drift_scores(shifted, distributions)
    assert scores["sales"]["drifted"] and not scores["prediction"]["drifted"]
    assert scores["sales"]["score"] > scores["sales"]["threshold"] > scores["prediction"]["score"]


def test_seasonal_reference_uses_the_latest_year_of_that_month(sales):
    reference = seasonal_reference(sales[sales["date"] < "2016-01-01"], pd.Timestamp("2016-02-01"))
    assert set(reference["date"].dt.strftime("%Y-%m")) == {"2015-02"}


def test_seasonal_reference_needs_the_month(sales):
    with pytest.raises(ValueError, match="no March"):
        seasonal_reference(sales[sales["date"].dt.month == 2], pd.Timestamp("2016-03-01"))


def test_naive_reference_samples_the_whole_window(sales):
    reference = naive_reference(sales, rows=500)
    assert len(reference) == 500
    assert reference["date"].dt.month.nunique() == 12
