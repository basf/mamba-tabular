from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ..base_models.embedding_classifier import BaseEmbeddingMambularClassifier
from ..utils.config import MambularConfig
from ..utils.preprocessor import Preprocessor
from ..utils.dataset import MambularDataModule, EmbeddingMambularDataset
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score


class EmbeddingMambularClassifier(BaseEstimator):
    """
    Provides an scikit-learn-like interface for the ProteinMambularClassifier, making it compatible with
    scikit-learn's utilities and workflow. This class encapsulates the PyTorch Lightning model, preprocessing,
    and data loading, offering methods for fitting, predicting, and probability estimation in a manner akin
    to scikit-learn's API.

    Parameters:
        **kwargs: Configuration parameters that can include both MambularConfig settings and preprocessing
                  options. Any unrecognized parameters are passed to the preprocessor.

    Attributes:
        config (MambularConfig): Configuration object for the model, storing architecture-specific parameters.
        preprocessor (Preprocessor): Object handling data preprocessing steps such as feature encoding and normalization.
        model (ProteinMambularClassifier): The underlying neural network model, instantiated during the `fit` method.

    Methods:
        fit: Fits the model to the provided dataset.
        predict: Predicts class labels for the provided data.
        predict_proba: Predicts class probabilities for the provided data.
        get_params: Returns the parameters of the classifier.
        set_params: Sets the parameters of the classifier.
    """

    def __init__(self, **kwargs):
        # Known config arguments
        config_arg_names = [
            "d_model",
            "n_layers",
            "dt_rank",
            "output_dimension",
            "pooling_method",
            "norm",
            "cls",
            "dt_min",
            "dt_max",
            "dropout",
            "bias",
            "weight_decay",
            "conv_bias",
            "d_state",
            "expand_factor",
            "d_conv",
            "dt_init",
            "dt_scale",
            "dt_init_floor",
        ]
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_arg_names}
        self.config = MambularConfig(**config_kwargs)

        # The rest are assumed to be preprocessor arguments
        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k not in config_arg_names
        }
        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.model = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
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
        Set the parameters of this estimator.

        Parameters:
        -----------
        **parameters : dict
            Estimator parameters.

        Returns:
        --------
        self : object
            Estimator instance.
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

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        val_size : float
            The proportion of the dataset to include in the validation split.
        random_state : int
            Controls the shuffling applied to the data before applying the split.

        Returns:
        --------
        X_train, X_val, y_train, y_val : arrays
            The split datasets.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )

        return X_train, X_val, y_train, y_val

    def preprocess_data(self, X_train, y_train, X_val, y_val, batch_size, shuffle):
        """
        Preprocess the training and validation data and create corresponding DataLoaders.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The training target values.
        X_val : array-like of shape (n_samples, n_features)
            The validation input samples.
        y_val : array-like of shape (n_samples,)
            The validation target values.
        batch_size : int
            Size of mini-batches for the DataLoader.
        shuffle : bool
            Whether to shuffle the training data before splitting into batches.

        Returns:
        --------
        data_module : MambularDataModule
            An instance of MambularDataModule containing training and validation DataLoaders.
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
            num_key = "num_" + str(
                key
            )  # Assuming numerical keys are prefixed with 'num_'
            if num_key in train_preprocessed_data:
                train_num_tensors.append(
                    torch.tensor(train_preprocessed_data[num_key], dtype=torch.float)
                )
            if num_key in val_preprocessed_data:
                val_num_tensors.append(
                    torch.tensor(val_preprocessed_data[num_key], dtype=torch.float)
                )

        train_labels = torch.tensor(y_train, dtype=torch.long)
        val_labels = torch.tensor(y_val, dtype=torch.long)

        # Create datasets
        train_dataset = EmbeddingMambularDataset(
            train_cat_tensors, train_num_tensors, train_labels, regression=False
        )
        val_dataset = EmbeddingMambularDataset(
            val_cat_tensors, val_num_tensors, val_labels, regression=False
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return MambularDataModule(train_dataloader, val_dataloader)

    def preprocess_test_data(self, X):
        """
        Preprocesses the test data and creates tensors for categorical and numerical features.

        Parameters:
        -----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            Test feature set.

        Returns:
        --------
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
            cat_key = "cat_" + str(
                key
            )  # Assuming categorical keys are prefixed with 'cat_'
            if cat_key in processed_data:
                cat_tensors.append(
                    torch.tensor(processed_data[cat_key], dtype=torch.long)
                )

            binned_key = "num_" + str(key)  # for binned features
            if binned_key in processed_data:
                cat_tensors.append(
                    torch.tensor(processed_data[binned_key], dtype=torch.long)
                )

        # Populate tensors for numerical features
        for key in self.num_feature_info:
            num_key = "num_" + str(
                key
            )  # Assuming numerical keys are prefixed with 'num_'
            if num_key in processed_data:
                num_tensors.append(
                    torch.tensor(processed_data[num_key], dtype=torch.float)
                )

        return cat_tensors, num_tensors

    def fit(
        self,
        X,
        y,
        val_size=0.2,
        X_val=None,
        y_val=None,
        max_epochs=100,
        random_state=101,
        batch_size=64,
        shuffle=True,
        patience=10,
        monitor="val_loss",
        mode="min",
        lr=1e-3,
        lr_patience=10,
        factor=0.75,
        weight_decay=0.025,
        raw_embeddings=False,
        seq_size=20,
        pca=False,
        reduced_dims=16,
        **trainer_kwargs
    ):
        """
        Fits the model to the given dataset.

        Parameters:
            X: Feature matrix for training, either as a pandas DataFrame or array-like.
            y: Target vector.
            val_size: Fraction of the data to use for validation if X_val is None.
            X_val: Feature matrix for validation.
            y_val: Target vector for validation.
            max_epochs: Maximum number of epochs for training.
            random_state: Seed for random number generators.
            batch_size: Size of batches for training and validation.
            shuffle: Whether to shuffle training data before each epoch.
            patience: Patience for early stopping based on val_loss.
            monitor: Metric to monitor for early stopping.
            mode: Mode for early stopping ('min' or 'max').
            lr: Learning rate for the optimizer.
            lr_patience: Patience for learning rate reduction.
            factor: Factor for learning rate reduction.
            weight_decay: Weight decay for the optimizer.
            raw_embeddings: Whether to use raw features or embeddings.
            seq_size: Sequence size for embeddings, relevant if raw_embeddings is False.
            **trainer_kwargs: Additional arguments for the PyTorch Lightning Trainer.

        Returns:
            self: The fitted estimator.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X_val:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)

        # Apply PCA if indicated
        if pca:
            pca_transformer = PCA(n_components=reduced_dims)
            X = pca_transformer.fit_transform(
                X
            )  # Fit and transform the PCA on the complete dataset
            if X_val is not None:
                X_val = pca_transformer.transform(
                    X_val
                )  # Transform validation data with the same PCA model

            raw_embeddings = True

        if not X_val:
            X_train, X_val, y_train, y_val = self.split_data(
                X, y, val_size, random_state
            )
        else:
            X_train = X
            y_train = y

        data_module = self.preprocess_data(
            X_train, y_train, X_val, y_val, batch_size, shuffle
        )

        if raw_embeddings:
            self.config.d_model = X.shape[1]

        num_classes = len(np.unique(y))

        self.model = BaseEmbeddingMambularClassifier(
            num_classes=num_classes,
            config=self.config,
            cat_feature_info=self.cat_feature_info,
            num_feature_info=self.num_feature_info,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
            raw_embeddings=raw_embeddings,
            seq_size=seq_size,
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
        trainer.fit(self.model, data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict(self, X):
        """
        Predict the class labels for the given input samples.

        Parameters:
        -----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.

        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels for each input sample.

        Notes:
        ------
        The method preprocesses the input data using the same preprocessor used during training,
        sets the model to evaluation mode, and then performs inference to predict the class labels.
        The predictions are converted from a PyTorch tensor to a NumPy array before being returned.
        """
        # Preprocess the data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if hasattr(self, "pca_transformer"):
            X = pd.DataFrame(self.pca_transformer.transform(X))

        cat_tensors, num_tensors = self.preprocess_test_data(X)
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
            logits = self.model(cat_tensors, num_tensors)
            predictions = torch.argmax(logits, dim=1)

        # Convert predictions to NumPy array and return
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input samples.

        Parameters:
        -----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples for which to predict class probabilities.

        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for each input sample.

        Notes:
        ------
        The method preprocesses the input data using the same preprocessor used during training,
        sets the model to evaluation mode, and then performs inference to predict the class probabilities.
        Softmax is applied to the logits to obtain probabilities, which are then converted from a PyTorch tensor
        to a NumPy array before being returned.
        """
        # Preprocess the data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cat_tensors, num_tensors = self.preprocess_test_data(X)
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
            logits = self.model(cat_tensors, num_tensors)
            probabilities = torch.softmax(logits, dim=1)

        # Convert probabilities to NumPy array and return
        return probabilities.cpu().numpy()

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on the given data using specified metrics.

        Parameters:
        -----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are tuples containing the metric function
            and a boolean indicating whether the metric requires probability scores (True) or class labels (False).

        Returns:
        --------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.

        Notes:
        ------
        This method uses either the `predict` or `predict_proba` method depending on the metric requirements.
        """
        # Ensure input is in the correct format
        if metrics is None:
            metrics = {"Accuracy": (accuracy_score, False)}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize dictionary to store results
        scores = {}

        # Generate class probabilities if any metric requires them
        if any(use_proba for _, use_proba in metrics.values()):
            probabilities = self.predict_proba(X)

        # Generate class labels if any metric requires them
        if any(not use_proba for _, use_proba in metrics.values()):
            predictions = self.predict(X)

        # Compute each metric
        for metric_name, (metric_func, use_proba) in metrics.items():
            if use_proba:
                scores[metric_name] = metric_func(y_true, probabilities)
            else:
                scores[metric_name] = metric_func(y_true, predictions)

        return scores