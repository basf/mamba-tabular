<div align="center">
  <img src="./docs/images/logo/mamba_tabular.jpg" width="400"/>


[![PyPI](https://img.shields.io/pypi/v/mambular)](https://pypi.org/project/mambular)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mambular)
[![docs build](https://readthedocs.org/projects/mambular/badge/?version=latest)](https://mambular.readthedocs.io/en/latest/?badge=latest)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mambular.readthedocs.io/en/latest/)
[![open issues](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/basf/mamba-tabular/issues)


[📘Documentation](https://mambular.readthedocs.io/en/latest/index.html) |
[🛠️Installation](https://mambular.readthedocs.io/en/latest/installation.html) |
[Models](https://mambular.readthedocs.io/en/latest/api/models/index.html) |
[🤔Report Issues](https://github.com/basf/mamba-tabular/issues)
</div>

<div style="text-align: center;">
    <h1>Mambular: Tabular Deep Learning (with Mamba)</h1>
</div>

Mambular is a Python library for tabular deep learning. It includes models that leverage the Mamba (State Space Model) architecture, as well as other popular models like TabTransformer, FTTransformer, and tabular ResNets. 

<h3> Table of Contents </h3>

- [🏃 Quickstart](#-quickstart)
- [📖 Introduction](#-introduction)
- [🤖 Models](#-models)
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [💻 Implement Your Own Model](#-implement-your-own-model)
- [License](#license)


# 🏃 Quickstart
Similar to any sklearn model, Mambular models can be fit as easy as this:

```python
from mambular.models import MambularClassifier
# Initialize and fit your model
model = MambularClassifier()

# X can be a dataframe or something that can be easily transformed into a pd.DataFrame as a np.array
model.fit(X, y, max_epochs=150, lr=1e-04)
```

# 📖 Introduction
Mambular is a Python package that brings the power of advanced deep learning architectures to tabular data, offering a suite of models for regression, classification, and distributional regression tasks. Designed with ease of use in mind, Mambular models adhere to scikit-learn's `BaseEstimator` interface, making them highly compatible with the familiar scikit-learn ecosystem. This means you can fit, predict, and evaluate using Mambular models just as you would with any traditional scikit-learn model, but with the added performance and flexibility of deep learning.


# 🤖 Models

| Model            | Description                                                                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Mambular`       | A sequential model using Mamba blocks [Gu and Dao](https://arxiv.org/pdf/2312.00752)  specifically designed for various tabular data tasks.         |
| `FTTransformer`  | A model leveraging transformer encoders, as introduced by [Gorishniy et al.](https://arxiv.org/abs/2106.11959), for tabular data.                   |
| `MLP`            | A classical Multi-Layer Perceptron (MLP) model for handling tabular data tasks.                                                                     |
| `ResNet`         | An adaptation of the ResNet architecture for tabular data applications.                                                                             |
| `TabTransformer` | A transformer-based model for tabular data introduced by [Huang et al.](https://arxiv.org/abs/2012.06678), enhancing feature learning capabilities. |
| `MambaTab`       | A tabular model using a Mamba-Block on a joint input representation described [here](https://arxiv.org/abs/2401.08867) . Not a sequential model.    |


All models are available for `regression`, `classification` and distributional regression, denoted by `LSS`.
Hence, they are available as e.g. `MambularRegressor`, `MambularClassifier` or `MambularLSS`


# 🛠️ Installation

Install Mambular using pip:
```sh
pip install mambular
```

# 🚀 Usage

<h2> Preprocessing </h2>

Mambular simplifies data preprocessing with a range of tools designed for easy transformation of tabular data.

<h3> Data Type Detection and Transformation </h3>

- **Ordinal & One-Hot Encoding**: Automatically transforms categorical data into numerical formats.
- **Binning**: Discretizes numerical features; can use decision trees for optimal binning.
- **Normalization & Standardization**: Scales numerical data appropriately.
- **Periodic Linear Encoding (PLE)**: Encodes periodicity in numerical data.
- **Quantile & Spline Transformations**: Applies advanced transformations to handle nonlinearity and distributional shifts.
- **Polynomial Features**: Generates polynomial and interaction terms to capture complex relationships.


<h2> Fit a Model </h2>
Fitting a model in mambular is as simple as it gets. All models in mambular are sklearn BaseEstimators. Thus the `.fit` method is implemented for all of them. Additionally, this allows for using all other sklearn inherent methods such as their built in hyperparameter optimization tools.

```python
from mambular.models import MambularClassifier
# Initialize and fit your model
model = MambularClassifier(
    d_model=64,
    n_layers=8,
    numerical_preprocessing="ple",
    n_bins=50
)

# X can be a dataframe or something that can be easily transformed into a pd.DataFrame as a np.array
model.fit(X, y, max_epochs=150, lr=1e-04)
```

Predictions are also easily obtained:
```python
# simple predictions
preds = model.predict(X)

# Predict probabilities
preds = model.predict_proba(X)
```


<h2> ⚖️ Distributional Regression with MambularLSS </h2>

MambularLSS allows you to model the full distribution of a response variable, not just its mean. This is crucial when understanding variability, skewness, or kurtosis is important. All Mambular models are available as distributional models.

<h3> Key Features of MambularLSS: </h3>

- **Full Distribution Modeling**: Predicts the entire distribution, not just a single value, providing richer insights.
- **Customizable Distribution Types**: Supports various distributions (e.g., Gaussian, Poisson, Binomial) for different data types.
- **Location, Scale, Shape Parameters**: Predicts key distributional parameters for deeper insights.
- **Enhanced Predictive Uncertainty**: Offers more robust predictions by modeling the entire distribution.

<h3> Available Distribution Classes: </h3>

- **normal**: For continuous data with a symmetric distribution.
- **poisson**: For count data within a fixed interval.
- **gamma**: For skewed continuous data, often used for waiting times.
- **beta**: For data bound between 0 and 1, like proportions.
- **dirichlet**: For multivariate data with correlated components.
- **studentt**: For data with heavier tails, useful with small samples.
- **negativebinom**: For over-dispersed count data.
- **inversegamma**: Often used as a prior in Bayesian inference.
- **categorical**: For data with more than two categories.
- **Quantile**: For quantile regression using the pinball loss.

These distribution classes make MambularLSS versatile in modeling various data types and distributions.


<h3> Getting Started with MambularLSS: </h3>

To integrate distributional regression into your workflow with `MambularLSS`, start by initializing the model with your desired configuration, similar to other Mambular models:

```python
from mambular.models import MambularLSS

# Initialize the MambularLSS model
model = MambularLSS(
    dropout=0.2,
    d_model=64,
    n_layers=8,
 
)

# Fit the model to your data
model.fit(
    X, 
    y, 
    max_epochs=150, 
    lr=1e-04, 
    patience=10,     
    family="normal" # define your distribution
    )

```


# 💻 Implement Your Own Model

Mambular allows users to easily integrate their custom models into the existing logic. This process is designed to be straightforward, making it simple to create a PyTorch model and define its forward pass. Instead of inheriting from `nn.Module`, you inherit from Mambular's `BaseModel`. Each Mambular model takes three main arguments: the number of classes (e.g., 1 for regression or 2 for binary classification), `cat_feature_info`, and `num_feature_info` for categorical and numerical feature information, respectively. Additionally, you can provide a config argument, which can either be a custom configuration or one of the provided default configs.

One of the key advantages of using Mambular is that the inputs to the forward passes are lists of tensors. While this might be unconventional, it is highly beneficial for models that treat different data types differently. For example, the TabTransformer model leverages this feature to handle categorical and numerical data separately, applying different transformations and processing steps to each type of data.

Here's how you can implement a custom model with Mambular:

1. **First, define your config:**  
   The configuration class allows you to specify hyperparameters and other settings for your model. This can be done using a simple dataclass.

   ```python
   from dataclasses import dataclass

   @dataclass
   class MyConfig:
       lr: float = 1e-04
       lr_patience: int = 10
       weight_decay: float = 1e-06
       lr_factor: float = 0.1
   ```

2. **Second, define your model:**  
   Define your custom model just as you would for an `nn.Module`. The main difference is that you will inherit from `BaseModel` and use the provided feature information to construct your layers. To integrate your model into the existing API, you only need to define the architecture and the forward pass.

   ```python
   from mambular.base_models import BaseModel
   import torch
   import torch.nn

   class MyCustomModel(BaseModel):
       def __init__(
           self,
           cat_feature_info,
           num_feature_info,
           num_classes: int = 1,
           config=None,
           **kwargs,
       ):
           super().__init__(**kwargs)
           self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

           input_dim = 0
           for feature_name, input_shape in num_feature_info.items():
               input_dim += input_shape
           for feature_name, input_shape in cat_feature_info.items():
               input_dim += 1 

           self.linear = nn.Linear(input_dim, num_classes)

       def forward(self, num_features, cat_features):
           x = num_features + cat_features
           x = torch.cat(x, dim=1)
           
           # Pass through linear layer
           output = self.linear(x)
           return output
   ```

3. **Leverage the Mambular API:**  
   You can build a regression, classification, or distributional regression model that can leverage all of Mambular's built-in methods by using the following:

   ```python
   from mambular.models import SklearnBaseRegressor

   class MyRegressor(SklearnBaseRegressor):
       def __init__(self, **kwargs):
           super().__init__(model=MyCustomModel, config=MyConfig, **kwargs)
   ```

4. **Train and evaluate your model:**  
   You can now fit, evaluate, and predict with your custom model just like with any other Mambular model. For classification or distributional regression, inherit from `SklearnBaseClassifier` or `SklearnBaseLSS` respectively.

   ```python
   regressor = MyRegressor(numerical_preprocessing="ple")
   regressor.fit(X_train, y_train, max_epochs=50)
   ```



# License

The entire codebase is under MIT license.
