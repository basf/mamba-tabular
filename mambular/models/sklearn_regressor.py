import lightning as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import warnings
import numpy as np
from ..base_models.regressor import BaseMambularRegressor
from ..utils.dataset import MambularDataModule, MambularDataset
from ..utils.preprocessor import Preprocessor
from ..utils.configs import DefaultMambularConfig


class MambularRegressor(BaseEstimator):
    """
    A regressor implemented using PyTorch Lightning that follows the scikit-learn API conventions.
    This class is designed to work with tabular data, offering a straightforward way to specify
    model configurations and preprocessing steps. It integrates seamlessly with scikit-learn's tools
    such as cross-validation and grid search.

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
    cat_cutoff: float or int, optional
        Indicates the cutoff after which integer values are treated as categorical. If float, it's treated as a percentage. If int, it's the maximum number of unique values for a column to be considered categorical. Default is 3%
    treat_all_integers_as_numerical : bool, optional
        If True, all integer columns will be treated as numerical, regardless of their unique value count or proportion. Default is False



    Attributes
    ----------
    config : DefaultConfig
        An object storing the configuration settings for the model.
    preprocessor : Preprocessor
        An object responsible for preprocessing the input data, such as encoding categorical variables and scaling numerical features.
    model : BaseMambularRegressor or None
        The underlying regression model, which is a PyTorch Lightning module. It is instantiated when the `fit` method is called.
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
            "cat_cutoff",
            "treat_all_integers_as_numerical",
        ]

        self.config_kwargs = {k: v for k, v in kwargs.items() if k in config_arg_names}
        self.config = DefaultMambularConfig(**self.config_kwargs)

        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k in preprocessor_arg_names
        }

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.model = None

        # Raise a warning if task is set to 'classification'
        if preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. MambularRegressor is designed for regression tasks.",
                UserWarning,
            )

    def get_params(self, deep=True):
        """
        Get parameters for this estimator. Overrides the BaseEstimator method.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns the parameters for this estimator and contained sub-objects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = self.config_kwargs  # Parameters used to initialize DefaultConfig

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
        Set the parameters of this estimator. Overrides the BaseEstimator method.

        Parameters
        ----------
        **parameters : dict
            Estimator parameters to be set.

        Returns
        -------
        self : object
            The instance with updated parameters.
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
        Splits the dataset into training and validation sets.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Input features.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        val_size : float
            The proportion of the dataset to include in the validation split.
        random_state : int
            Controls the shuffling applied to the data before applying the split.


        Returns
        -------
        X_train, X_val, y_train, y_val : arrays
            The split datasets.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )

        return X_train, X_val, y_train, y_val

    def preprocess_data(self, X_train, y_train, X_val, y_val, batch_size, shuffle):
        """
        Preprocesses the training and validation data, and creates DataLoaders for them.

        Parameters
        ----------
        X_train : DataFrame or array-like, shape (n_samples_train, n_features)
            Training feature set.
        y_train : array-like, shape (n_samples_train,)
            Training target values.
        X_val : DataFrame or array-like, shape (n_samples_val, n_features)
            Validation feature set.
        y_val : array-like, shape (n_samples_val,)
            Validation target values.
        batch_size : int
            Size of batches for the DataLoader.
        shuffle : bool
            Whether to shuffle the training data in the DataLoader.


        Returns
        -------
        data_module : MambularDataModule
            An instance of MambularDataModule containing the training and validation DataLoaders.
        """
        self.preprocessor.fit(
            pd.concat([X_train, X_val], axis=0).reset_index(drop=True),
            np.concatenate((y_train, y_val), axis=0),
        )
        train_preprocessed_data = self.preprocessor.transform(X_train)
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
        Preprocesses the test data and creates tensors for categorical and numerical features.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            Test feature set.


        Returns
        -------
        cat_tensors : list of Tensors
            List of tensors for each categorical feature.
        num_tensors : list of Tensors
            List of tensors for each numerical feature.
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
        Trains the regression model using the provided training data. Optionally, a separate validation set can be used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None. Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        max_epochs : int, default=100
            Maximum number of epochs for training.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before early stopping.
        monitor : str, default="val_loss"
            The metric to monitor for early stopping.
        mode : str, default="min"
            Whether the monitored metric should be minimized (`min`) or maximized (`max`).
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) coefficient.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted regressor.
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

        self.data_module = self.preprocess_data(
            X_train, y_train, X_val, y_val, batch_size, shuffle
        )

        self.model = BaseMambularRegressor(
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
            monitor="val_loss",  # Adjust according to your validation metric
            mode="min",
            save_top_k=1,
            dirpath="model_checkpoints",  # Specify the directory to save checkpoints
            filename="best_model",
        )

        # Initialize the trainer and train the model
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            **trainer_kwargs
        )
        trainer.fit(self.model, self.data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["state_dict"])

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
            predictions = self.model(num_features=num_tensors, cat_features=cat_tensors)

        # Convert predictions to NumPy array and return
        return predictions.cpu().numpy()

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are the metric functions.


        Notes
        -----
        This method uses the `predict` method to generate predictions and computes each metric.


        Examples
        --------
        >>> from sklearn.metrics import mean_squared_error, r2_score
        >>> from sklearn.model_selection import train_test_split
        >>> from mambular.models import MambularRegressor
        >>> metrics = {
        ...     'Mean Squared Error': mean_squared_error,
        ...     'R2 Score': r2_score
        ... }
        >>> # Assuming 'X_test' and 'y_test' are your test dataset and labels
        >>> # Evaluate using the specified metrics
        >>> results = regressor.evaluate(X_test, y_test, metrics=metrics)


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.
        """
        if metrics is None:
            metrics = {"Mean Squared Error": mean_squared_error}

        # Ensure input is in the correct format
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Generate predictions using the trained model
        predictions = self.predict(X)

        # Initialize dictionary to store results
        scores = {}

        # Compute each metric
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = metric_func(y_true, predictions)

        return scores
