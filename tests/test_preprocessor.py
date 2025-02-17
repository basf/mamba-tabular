import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from mambular.preprocessing import Preprocessor


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "numerical": np.random.randn(100),
            "categorical": np.random.choice(["A", "B", "C"], size=100),
            "integer": np.random.randint(0, 5, size=100),
        }
    )


@pytest.fixture
def sample_target():
    return np.random.randn(100)


@pytest.fixture(
    params=[
        "ple",
        "binning",
        "one-hot",
        "standardization",
        "minmax",
        "quantile",
        "polynomial",
        "robust",
        "splines",
        "yeo-johnson",
        "box-cox",
        "rbf",
        "sigmoid",
        "none",
    ]
)
def preprocessor(request):
    return Preprocessor(
        numerical_preprocessing=request.param, categorical_preprocessing="one-hot"
    )


def test_preprocessor_initialization(preprocessor):
    assert preprocessor.numerical_preprocessing in [
        "ple",
        "binning",
        "one-hot",
        "standardization",
        "minmax",
        "quantile",
        "polynomial",
        "robust",
        "splines",
        "yeo-johnson",
        "box-cox",
        "rbf",
        "sigmoid",
        "none",
    ]
    assert preprocessor.categorical_preprocessing == "one-hot"
    assert not preprocessor.fitted


def test_preprocessor_fit(preprocessor, sample_data, sample_target):
    preprocessor.fit(sample_data, sample_target)
    assert preprocessor.fitted
    assert preprocessor.column_transformer is not None


def test_preprocessor_transform(preprocessor, sample_data, sample_target):
    preprocessor.fit(sample_data, sample_target)
    transformed = preprocessor.transform(sample_data)
    assert isinstance(transformed, dict)
    assert len(transformed) > 0


def test_preprocessor_fit_transform(preprocessor, sample_data, sample_target):
    transformed = preprocessor.fit_transform(sample_data, sample_target)
    assert isinstance(transformed, dict)
    assert len(transformed) > 0


def test_preprocessor_get_params(preprocessor):
    params = preprocessor.get_params()
    assert "n_bins" in params
    assert "numerical_preprocessing" in params


def test_preprocessor_set_params(preprocessor):
    preprocessor.set_params(n_bins=128)
    assert preprocessor.n_bins == 128


def test_transform_before_fit_raises_error(preprocessor, sample_data):
    with pytest.raises(NotFittedError):
        preprocessor.transform(sample_data)


def test_get_feature_info(preprocessor, sample_data, sample_target):
    preprocessor.fit(sample_data, sample_target)
    numerical_info, categorical_info, embedding_info = preprocessor.get_feature_info(
        verbose=False
    )
    assert isinstance(numerical_info, dict)
    assert isinstance(categorical_info, dict)
    assert isinstance(embedding_info, dict)
