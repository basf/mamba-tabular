from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ..base_models.embedding_regressor import BaseEmbeddingMambularRegressor
from ..utils.config import MambularConfig
from ..utils.preprocessor import Preprocessor
from ..utils.dataset import MambularDataModule, EmbeddingMambularDataset
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


class EmbeddingMambularRegressor(BaseEstimator):
    """
    An sklearn-like interface for the ProteinMambularRegressor, making it compatible with sklearn's utilities
    and workflows. This class wraps the PyTorch Lightning model and preprocessor, providing methods for fitting,
    predicting, and setting/getting parameters in a way that mimics sklearn's API.

    Parameters:
        **kwargs: Keyword arguments that can include both configuration parameters for the MambularConfig and
                  parameters for the preprocessor.

    Attributes:
        config (MambularConfig): Configuration object containing model-specific parameters.
        preprocessor (Preprocessor): Preprocessor object for data preprocessing steps.
        model (ProteinMambularRegressor): The neural network model, initialized after the `fit` method is called.
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

        if not "numerical_preprocessing" in preprocessor_kwargs.keys():
            preprocessor_kwargs["numerical_preprocessing"] = "normalization"
        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.model = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator. Overrides the BaseEstimator method.

        Parameters:
        -----------
        deep : bool, default=True
            If True, returns the parameters for this estimator and contained sub-objects that are estimators.

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
        Set the parameters of this estimator. Overrides the BaseEstimator method.

        Parameters:
        -----------
        **parameters : dict
            Estimator parameters to be set.

        Returns:
        --------
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

        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Input features.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
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
        Preprocesses the training and validation data, and creates DataLoaders for them.

        Parameters:
        -----------
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

        Returns:
        --------
        data_module : MambularDataModule
            An instance of MambularDataModule containing the training and validation DataLoaders.
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

        train_labels = torch.tensor(y_train, dtype=torch.float)
        val_labels = torch.tensor(y_val, dtype=torch.float)

        # Create datasets
        train_dataset = EmbeddingMambularDataset(
            train_cat_tensors, train_num_tensors, train_labels
        )
        val_dataset = EmbeddingMambularDataset(
            val_cat_tensors, val_num_tensors, val_labels
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
        **trainer_kwargs
    ):
        """
        Fits the ProteinMambularRegressor model to the training data.

        Parameters:
            X (array-like or DataFrame): The training input samples.
            y (array-like): The target values (class labels for classification, real numbers for regression).
            val_size (float, optional): The proportion of the dataset to include in the validation split if `X_val` is not provided.
            X_val (array-like or DataFrame, optional): The validation input samples.
            y_val (array-like, optional): The validation target values.
            max_epochs (int, optional): The maximum number of epochs for training.
            random_state (int, optional): The seed used by the random number generator.
            batch_size (int, optional): Size of the batches for training.
            shuffle (bool, optional): Whether to shuffle the training data.
            patience (int, optional): Patience for early stopping.
            monitor (str, optional): Quantity to be monitored for early stopping.
            mode (str, optional): One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing.
            lr (float, optional): Learning rate for the optimizer.
            lr_patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced.
            factor (float, optional): Factor by which the learning rate will be reduced.
            weight_decay (float, optional): Weight decay coefficient for regularization in the optimizer.
            raw_embeddings (bool, optional): Whether to use raw numerical features directly or to process them into embeddings.
            seq_size (int, optional): The sequence size for processing numerical features when not using raw embeddings.
            **trainer_kwargs: Additional keyword arguments for the PyTorch Lightning Trainer.

        Returns:
            self: Returns an instance of self.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X_val:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)

        # Apply PCA if indicated
        if pca:
            self.pca_transformer = PCA(n_components=seq_size)
            X = pd.DataFrame(
                self.pca_transformer.fit_transform(X)
            )  # Fit and transform the PCA on the complete dataset
            if X_val is not None:
                X_val = pd.DataFrame(
                    self.pca_transformer.transform(X_val)
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

        self.model = BaseEmbeddingMambularRegressor(
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
        Predicts target values for the given input samples.

        Parameters:
        -----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.

        Returns:
        --------
        predictions : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values.
        """
        # Preprocess the data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if hasattr(self, "pca_transformer"):
            X = pd.DataFrame(self.pca_transformer.transform(X))

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

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on the given data using specified metrics.

        Example:
            metrics = {
                'Mean Squared Error': mean_squared_error,
                'R2 Score': r2_score
            }

            # Assuming 'X_test' and 'y_test' are your test dataset and labels
            # Evaluate using the specified metrics
            results = regressor.evaluate(X_test, y_test, metrics=metrics)

        Parameters:
        -----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are the metric functions.

        Returns:
        --------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.

        Notes:
        ------
        This method uses the `predict` method to generate predictions and computes each metric.
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
