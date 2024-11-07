import lightning as pl
import pandas as pd
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import warnings
from .forecast_wrapper import ForecastingTaskModel
from ...data_utils.forecast_datamodule import ForecastMambularDataModule
from ...preprocessing import Preprocessor
from lightning.pytorch.callbacks import ModelSummary
from skopt import gp_minimize
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import warnings
from ...utils.config_mapper import (
    get_search_space,
    activation_mapper,
    round_to_nearest_16,
)


class SklearnBaseTimeSeriesForecaster(BaseEstimator):
    def __init__(self, model, config, **kwargs):
        self.preprocessor_arg_names = [
            "n_bins",
            "numerical_preprocessing",
            "use_decision_tree_bins",
            "binning_strategy",
            "task",
            "cat_cutoff",
            "treat_all_integers_as_numerical",
            "knots",
            "degree",
        ]

        time_series_args = ["time_steps", "max_seq_length", "forecast_horizon"]

        self.config_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in self.preprocessor_arg_names
            and not k.startswith("optimizer")
            and k not in time_series_args
        }
        self.config = config(**self.config_kwargs)

        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k in self.preprocessor_arg_names
        }

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.base_model = model
        self.forecast_model = None
        self.built = False

        if preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. The Forecaster is designed for time series forecasting tasks.",
                UserWarning,
            )

        self.optimizer_type = kwargs.get("optimizer_type", "Adam")

        self.optimizer_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in ["lr", "weight_decay", "patience", "lr_patience", "optimizer_type"]
            and k.startswith("optimizer_")
        }

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        params.update(self.config_kwargs)

        if deep:
            preprocessor_params = {
                "preprocessor__" + key: value
                for key, value in self.preprocessor.get_params().items()
            }
            params.update(preprocessor_params)

        return params

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **parameters : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        config_params = {
            k: v for k, v in parameters.items() if not k.startswith("preprocessor__")
        }
        preprocessor_params = {
            k.split("__")[1]: v
            for k, v in parameters.items()
            if k.startswith("preprocessor__")
        }

        if config_params:
            self.config_kwargs.update(config_params)
            if self.config is not None:
                for key, value in config_params.items():
                    setattr(self.config, key, value)
            else:
                self.config = self.config_class(**self.config_kwargs)

        if preprocessor_params:
            self.preprocessor.set_params(**preprocessor_params)

        return self

    def build_model(
        self,
        data,
        val_size: float = 0.2,
        random_state: int = 101,
        batch_size: int = 128,
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        dataloader_kwargs={},
    ):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        self.data_module = ForecastMambularDataModule(
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            time_steps=self.config.time_steps,
            forecast_horizon=self.config.forecast_horizon,
            data=data,
            val_size=val_size,
            random_state=random_state,
            **dataloader_kwargs,
        )

        self.data_module.preprocess_data()

        self.forecast_model = ForecastingTaskModel(
            model_class=self.base_model,
            config=self.config,
            cat_feature_info=self.data_module.cat_feature_info,
            num_feature_info=self.data_module.num_feature_info,
            num_classes=self.config.forecast_horizon,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
            optimizer_type=self.optimizer_type,
            optimizer_args=self.optimizer_kwargs,
        )

        self.built = True

        return self

    def fit(
        self,
        data,
        val_size: float = 0.2,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        checkpoint_path="model_checkpoints",
        dataloader_kwargs={},
        rebuild=True,
        **trainer_kwargs,
    ):
        if rebuild:
            self.build_model(
                data,
                val_size=val_size,
                batch_size=batch_size,
                random_state=random_state,
                lr=lr,
                lr_patience=lr_patience,
                factor=factor,
                weight_decay=weight_decay,
                dataloader_kwargs=dataloader_kwargs,
            )

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,
            filename="best_model",
        )

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                early_stop_callback,
                checkpoint_callback,
                ModelSummary(max_depth=2),
            ],
            **trainer_kwargs,
        )
        self.trainer.fit(self.forecast_model, self.data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.forecast_model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict(self, X):
        """
        Predicts target values for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.

        Returns
        -------
        predictions : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values.
        """
        # Ensure model and data module are initialized
        if self.forecast_model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        # Preprocess and set up test data
        self.data_module.preprocess_test_data(X)
        self.data_module.setup("test")
        test_loader = self.data_module.test_dataloader()

        # Perform predictions on the test DataLoader
        predictions = self.trainer.predict(self.forecast_model, test_loader)

        # Concatenate predictions and convert to NumPy array
        return torch.cat(predictions).cpu().numpy()

    def evaluate(self, X, iterative=False, horizons=[1]):
        self.data_module.preprocess_test_data(X)
        self.data_module.setup("test")
        test_loader = self.data_module.test_dataloader()

        self.forecast_model.set_preprocessor(self.data_module.get_preprocessor())
        self.forecast_model.iterative = iterative
        self.forecast_model.horizons = horizons

        results = self.trainer.test(self.forecast_model, test_loader)[0]

        rmse = np.sqrt(
            mean_squared_error(
                np.array(self.forecast_model.test_predictions).squeeze(-1),
                np.array(self.forecast_model.test_targets).squeeze(-1),
            )
        )

        results["complete_rmse"] = rmse

        return results

    def optimize_hparams(
        self,
        data,
        val_size=0.2,
        time=100,
        max_epochs=200,
        prune_by_epoch=True,
        prune_epoch=5,
        **optimize_kwargs,
    ):
        param_names, param_space = get_search_space(self.config)

        self.fit(data, val_size=val_size, max_epochs=max_epochs)
        best_val_loss = float("inf")

        def _objective(hyperparams):
            nonlocal best_val_loss

            head_layer_sizes = []

            for key, param_value in zip(param_names, hyperparams):
                if key == "head_layer_size_length":
                    head_layer_size_length = param_value
                elif key.startswith("head_layer_size_"):
                    head_layer_sizes.append(round_to_nearest_16(param_value))
                else:
                    field_type = self.config.__dataclass_fields__[key].type
                    if field_type == callable and isinstance(param_value, str):
                        setattr(self.config, key, activation_mapper[param_value])
                    else:
                        setattr(self.config, key, param_value)

            if head_layer_size_length is not None:
                setattr(
                    self.config,
                    "head_layer_sizes",
                    head_layer_sizes[:head_layer_size_length],
                )

            self.build_model(
                data, val_size=val_size, lr=self.config.lr, **optimize_kwargs
            )

            self.forecast_model.early_pruning_threshold = (
                best_val_loss * 1.5 if not prune_by_epoch else None
            )
            self.forecast_model.pruning_epoch = prune_epoch

            self.fit(data, val_size=val_size, max_epochs=max_epochs, rebuild=False)

            val_loss = self.trainer.validate(self.forecast_model, self.data_module)[0][
                "val_loss"
            ]

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            return val_loss

        result = gp_minimize(_objective, param_space, n_calls=time, random_state=42)

        best_hparams = result.x
        if "head_layer_sizes" in self.config.__dataclass_fields__:
            head_layer_sizes = []

        for key, param_value in zip(param_names, best_hparams):
            if key.startswith("head_layer_size_"):
                head_layer_sizes.append(round_to_nearest_16(param_value))
            else:
                field_type = self.config.__dataclass_fields__[key].type
                if field_type == callable and isinstance(param_value, str):
                    setattr(self.config, key, activation_mapper[param_value])
                else:
                    setattr(self.config, key, param_value)

        if head_layer_sizes:
            setattr(self.config, "head_layer_sizes", head_layer_sizes)

        print("Best hyperparameters found:", best_hparams)

        return best_hparams
