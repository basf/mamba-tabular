import lightning as pl
import numpy as np
import warnings
import pandas as pd
import properscoring as ps
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..base_models.distributional import BaseMambularLSS
from ..utils.configs import DefaultMambularConfig
from ..utils.dataset import MambularDataModule, MambularDataset
from ..utils.distributional_metrics import (
    beta_brier_score,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)
from ..utils.preprocessor import Preprocessor


class MambularLSS(BaseEstimator):
    """
    MambularLSS is a machine learning estimator that is designed for structured data,
    incorporating both preprocessing and a deep learning model. The estimator
    integrates configurable components for data preprocessing and the neural network model,
    facilitating end-to-end training and prediction workflows.

    The initialization of this class separates configuration arguments for the model and
    the preprocessor, allowing for flexible adjustment of parameters.

    Parameters
    ----------
    # configuration parameters
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-4.
    lr_patience : int, optional
        Number of epochs with no improvement on the validation loss to wait before reducing the learning rate. Default is 10.
    weight_decay : float, optional
        Weight decay (L2 penalty) coefficient. Default is 1e-6.
    lr_factor : float, optional
        Factor by which the learning rate will be reduced. Default is 0.1.
    d_model : int, optional
        Dimension of the model. Default is 64.
    n_layers : int, optional
        Number of layers. Default is 8.
    expand_factor : int, optional
        Expansion factor. Default is 2.
    bias : bool, optional
        Whether to use bias. Default is False.
    d_conv : int, optional
        Dimension of the convolution. Default is 16.
    conv_bias : bool, optional
        Whether to use bias in the convolution. Default is True.
    dropout : float, optional
        Dropout rate in the mamba blocks. Default is 0.05.
    dt_rank : str, optional
        Rank of the time dimension. Default is "auto".
    d_state : int, optional
        State dimension. Default is 16.
    dt_scale : float, optional
        Scale of the time dimension. Default is 1.0.
    dt_init : str, optional
        Initialization method for the time dimension. Default is "random".
    dt_max : float, optional
        Maximum value for the time dimension. Default is 0.1.
    dt_min : float, optional
        Minimum value for the time dimension. Default is 1e-3.
    dt_init_floor : float, optional
        Floor value for the time dimension initialization. Default is 1e-4.
    norm : str, optional
        Normalization method. Default is 'RMSNorm'.
    activation : callable, optional
        Activation function. Default is nn.SELU().
    num_embedding_activation : callable, optional
        Activation function for numerical embeddings. Default is nn.Identity().
    head_layer_sizes : list, optional
        Sizes of the layers in the head. Default is [64, 64, 32].
    head_dropout : float, optional
        Dropout rate for the head. Default is 0.5.
    head_skip_layers : bool, optional
        Whether to use skip layers in the head. Default is False.
    head_activation : callable, optional
        Activation function for the head. Default is nn.SELU().
    head_use_batch_norm : bool, optional
        Whether to use batch normalization in the head. Default is False.

    # Preprocessor Parameters
    n_bins : int, optional
        The number of bins to use for numerical feature binning. Default is 50.
    numerical_preprocessing : str, optional
        The preprocessing strategy for numerical features. Default is 'ple'.
    use_decision_tree_bins : bool, optional
        If True, uses decision tree regression/classification to determine optimal bin edges for numerical feature binning. Default is False.
    binning_strategy : str, optional
        Defines the strategy for binning numerical features. Default is 'uniform'.
    task : str, optional
        Indicates the type of machine learning task ('regression' or 'classification'). Default is 'regression'.



    Attributes
    ----------
    config : MambularConfig
        Configuration object that holds model-specific settings.
    preprocessor : Preprocessor
        Preprocessor object for handling feature preprocessing like normalization and encoding.
    model : BaseMambularClassifier or None
        The underlying PyTorch Lightning model, instantiated upon calling the `fit` method.
    """

    def __init__(self, **kwargs):
        # Known config arguments
        config_arg_names = [
            "lr",
            "lr_patience",
            "weight_decay",
            "lr_factor",
            "d_model",
            "n_layers",
            "expand_factor",
            "bias",
            "d_conv",
            "conv_bias",
            "dropout",
            "dt_rank",
            "d_state",
            "dt_scale",
            "dt_init",
            "dt_max",
            "dt_min",
            "dt_init_floor",
            "norm",
            "activation",
            "num_embedding_activation",
            "head_layer_sizes",
            "head_dropout",
            "head_skip_layers",
            "head_activation",
            "head_use_batch_norm",
        ]

        preprocessor_arg_names = [
            "n_bins",
            "numerical_preprocessing",
            "use_decision_tree_bins",
            "binning_strategy",
            "task",
        ]
        self.config_kwargs = {k: v for k, v in kwargs.items() if k in config_arg_names}
        self.config = DefaultMambularConfig(**self.config_kwargs)

        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k in preprocessor_arg_names
        }
        # Raise a warning if task is set to 'classification'
        if preprocessor_kwargs.get("task") == "regression":
            warnings.warn(
                "The task in preprocessing binning is set to 'regression'. Make sure that this is correct for your distributional family ",
                UserWarning,
            )

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.model = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator, optionally including parameters from nested components
        like the preprocessor.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of nested components.


        Returns
        -------
        dict
            A dictionary mapping parameter names to their values. For nested components,
            parameter names are prefixed accordingly (e.g., 'preprocessor__<param_name>').
        """

        params = self.config_kwargs  # Parameters used to initialize MambularConfig

        # If deep=True, include parameters from nested components like preprocessor
        if deep:
            # Assuming Preprocessor has a get_params method
            preprocessor_params = {
                "preprocessor__" + key: value
                for key, value in self.preprocessor.get_params().items()
            }
            params.update(preprocessor_params)

        return params

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator, allowing for modifications to both the configuration
        and preprocessor parameters. Parameters not recognized as configuration arguments are
        assumed to be preprocessor arguments.

        Parameters
        ----------
            **parameters: Arbitrary keyword arguments where keys are parameter names and values
                          are the new parameter values.


        Returns
        -------
            self: This instance with updated parameters.
        """
        # Update config_kwargs with provided parameters
        valid_config_keys = self.config_kwargs.keys()
        config_updates = {k: v for k, v in parameters.items() if k in valid_config_keys}
        self.config_kwargs.update(config_updates)

        # Update the config object
        for key, value in config_updates.items():
            setattr(self.config, key, value)

        # Handle preprocessor parameters (prefixed with 'preprocessor__')
        preprocessor_params = {
            k.split("__")[1]: v
            for k, v in parameters.items()
            if k.startswith("preprocessor__")
        }
        if preprocessor_params:
            # Assuming Preprocessor has a set_params method
            self.preprocessor.set_params(**preprocessor_params)

        return self

    def split_data(self, X, y, val_size, random_state):
        """
        Split the dataset into training and validation sets.

        Parameters
        ----------
        X : array-like
            Features of the dataset.
        y : array-like
            Target values.
        val_size : float
            The proportion of the dataset to include in the validation split.
        random_state : int, optional
            The seed used by the random number generator for reproducibility.


        Returns
        -------
        tuple
            A tuple containing split datasets (X_train, X_val, y_train, y_val).
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )

        return X_train, X_val, y_train, y_val

    def preprocess_data(self, X_train, y_train, X_val, y_val, batch_size, shuffle):
        """
        Preprocess the training and validation data, fit the preprocessor on the training data,
        and transform both training and validation data. This method also initializes tensors
        for categorical and numerical features and labels, and prepares DataLoader objects for
        both datasets.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training target values.
        X_val : array-like
            Validation features.
        y_val : array-like
            Validation target values.
        batch_size : int
            Batch size for DataLoader objects.
        shuffle : bool
            Whether to shuffle the training data in the DataLoader.


        Returns
        -------
        MambularDataModule
            An object containing DataLoaders for training and validation datasets.
        """
        train_preprocessed_data = self.preprocessor.fit_transform(X_train, y_train)
        val_preprocessed_data = self.preprocessor.transform(X_val)

        # Update feature info based on the actual processed data
        (
            self.cat_feature_info,
            self.num_feature_info,
        ) = self.preprocessor.get_feature_info()

        # Initialize lists for tensors
        train_cat_tensors = []
        train_num_tensors = []
        val_cat_tensors = []
        val_num_tensors = []

        # Populate tensors for categorical features, if present in processed data
        for key in self.cat_feature_info:
            cat_key = "cat_" + key  # Assuming categorical keys are prefixed with 'cat_'
            if cat_key in train_preprocessed_data:
                train_cat_tensors.append(
                    torch.tensor(train_preprocessed_data[cat_key], dtype=torch.long)
                )
            if cat_key in val_preprocessed_data:
                val_cat_tensors.append(
                    torch.tensor(val_preprocessed_data[cat_key], dtype=torch.long)
                )

            binned_key = "num_" + key  # for binned features
            if binned_key in train_preprocessed_data:
                train_cat_tensors.append(
                    torch.tensor(train_preprocessed_data[binned_key], dtype=torch.long)
                )

            if binned_key in val_preprocessed_data:
                val_cat_tensors.append(
                    torch.tensor(val_preprocessed_data[binned_key], dtype=torch.long)
                )

        # Populate tensors for numerical features, if present in processed data
        for key in self.num_feature_info:
            num_key = "num_" + key  # Assuming numerical keys are prefixed with 'num_'
            if num_key in train_preprocessed_data:
                train_num_tensors.append(
                    torch.tensor(train_preprocessed_data[num_key], dtype=torch.float32)
                )
            if num_key in val_preprocessed_data:
                val_num_tensors.append(
                    torch.tensor(val_preprocessed_data[num_key], dtype=torch.float32)
                )

        train_labels = torch.tensor(y_train, dtype=torch.float32)
        val_labels = torch.tensor(y_val, dtype=torch.float32)

        # Create datasets
        train_dataset = MambularDataset(
            train_cat_tensors, train_num_tensors, train_labels
        )
        val_dataset = MambularDataset(val_cat_tensors, val_num_tensors, val_labels)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return MambularDataModule(train_dataloader, val_dataloader)

    def preprocess_test_data(self, X):
        """
        Preprocess test data using the fitted preprocessor. This method prepares tensors for
        categorical and numerical features based on the preprocessed test data.

        Parameters
        ----------
        X : array-like
            Test features to preprocess.


        Returns
        -------
        tuple
            A tuple containing lists of tensors for categorical and numerical features.
        """
        processed_data = self.preprocessor.transform(X)

        # Initialize lists for tensors
        cat_tensors = []
        num_tensors = []

        # Populate tensors for categorical features
        for key in self.cat_feature_info:
            cat_key = "cat_" + key  # Assuming categorical keys are prefixed with 'cat_'
            if cat_key in processed_data:
                cat_tensors.append(
                    torch.tensor(processed_data[cat_key], dtype=torch.long)
                )

            binned_key = "num_" + key  # for binned features
            if binned_key in processed_data:
                cat_tensors.append(
                    torch.tensor(processed_data[binned_key], dtype=torch.long)
                )

        # Populate tensors for numerical features
        for key in self.num_feature_info:
            num_key = "num_" + key  # Assuming numerical keys are prefixed with 'num_'
            if num_key in processed_data:
                num_tensors.append(
                    torch.tensor(processed_data[num_key], dtype=torch.float32)
                )

        return cat_tensors, num_tensors

    def fit(
        self,
        X,
        y,
        family,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        **trainer_kwargs
    ):
        """
        Fits the model to the provided data, using the specified loss distribution family for the prediction task.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            Training features.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values for training.
        family : str
            The name of the distribution family to use for the loss function. Examples include 'normal' for regression tasks.
        val_size : float, default=0.2
            Proportion of the dataset to include in the validation split if `X_val` is None.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            Validation features. If provided, `X` and `y` are not split.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            Validation target values. Required if `X_val` is provided.
        max_epochs : int, default=100
            Maximum number of epochs for training.
        random_state : int, default=101
            Seed used by the random number generator for shuffling the data.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=10
            Number of epochs with no improvement on the validation metric to wait before early stopping.
        monitor : str, default="val_loss"
            The metric to monitor for early stopping.
        mode : str, default="min"
            In 'min' mode, training will stop when the quantity monitored has stopped decreasing;
            in 'max' mode, it will stop when the quantity monitored has stopped increasing.
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation metric to wait before reducing the learning rate.
        factor : float, default=0.75
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) parameter.
        **trainer_kwargs : dict
            Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted estimator.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X_val:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)

        if not X_val:
            X_train, X_val, y_train, y_val = self.split_data(
                X, y, val_size, random_state
            )

        data_module = self.preprocess_data(
            X_train, y_train, X_val, y_val, batch_size, shuffle
        )

        self.model = BaseMambularLSS(
            family=family,
            config=self.config,
            cat_feature_info=self.cat_feature_info,
            num_feature_info=self.num_feature_info,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
        )

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="model_checkpoints",
            filename="best_model",
        )

        # Initialize the trainer and train the model
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            **trainer_kwargs,
        )
        trainer.fit(self.model, data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict(self, X):
        """
        Predicts target values for the given input samples using the fitted model.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.


        Returns
        -------
        predictions : ndarray, shape (n_samples,) or (n_samples, n_distributional_parameters)
            The predicted target values.
        """
        # Preprocess the data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        device = next(self.model.parameters()).device
        cat_tensors, num_tensors = self.preprocess_test_data(X)
        if isinstance(cat_tensors, list):
            cat_tensors = [tensor.to(device) for tensor in cat_tensors]
        else:
            cat_tensors = cat_tensors.to(device)

        if isinstance(num_tensors, list):
            num_tensors = [tensor.to(device) for tensor in num_tensors]
        else:
            num_tensors = num_tensors.to(device)

        # Set the model to evaluation mode
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            predictions = self.model(cat_tensors, num_tensors)

        # Convert predictions to NumPy array and return
        return predictions.cpu().numpy()

    def evaluate(self, X, y_true, metrics=None, distribution_family=None):
        """
        Evaluate the model on the given data using specified metrics tailored to the distribution type.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            Input samples.
        y_true : DataFrame or array-like, shape (n_samples,) or (n_samples, n_outputs)
            True target values.
        metrics : dict, optional
            A dictionary where keys are metric names and values are the metric functions.
            If None, default metrics based on the detected or specified distribution_family are used.
        distribution_family : str, optional
            Specifies the distribution family the model is predicting for. If None, it will attempt to infer based
            on the model's settings.


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.
        """
        # Infer distribution family from model settings if not provided
        if distribution_family is None:
            distribution_family = getattr(self.model, "distribution_family", "normal")

        # Setup default metrics if none are provided
        if metrics is None:
            metrics = self.get_default_metrics(distribution_family)

        # Make predictions
        predictions = self.predict(X)

        # Initialize dictionary to store results
        scores = {}

        # Compute each metric
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = metric_func(y_true, predictions)

        return scores

    def get_default_metrics(self, distribution_family):
        """
        Provides default metrics based on the distribution family.

        Parameters
        ----------
        distribution_family : str
            The distribution family for which to provide default metrics.


        Returns
        -------
        metrics : dict
            A dictionary of default metric functions.
        """
        default_metrics = {
            "normal": {
                "MSE": lambda y, pred: mean_squared_error(y, pred[:, 0]),
                "CRPS": lambda y, pred: np.mean(
                    [
                        ps.crps_gaussian(y[i], mu=pred[i, 0], sig=np.sqrt(pred[i, 1]))
                        for i in range(len(y))
                    ]
                ),
            },
            "poisson": {"Poisson Deviance": poisson_deviance},
            "gamma": {"Gamma Deviance": gamma_deviance},
            "beta": {"Brier Score": beta_brier_score},
            "dirichlet": {"Dirichlet Error": dirichlet_error},
            "studentt": {"Student-T Loss": student_t_loss},
            "negativebinom": {"Negative Binomial Deviance": negative_binomial_deviance},
            "inversegamma": {"Inverse Gamma Loss": inverse_gamma_loss},
            "categorical": {"Accuracy": accuracy_score},
        }
        return default_metrics.get(distribution_family, {})
