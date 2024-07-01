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

# Mambular: Tabular Deep Learning with Mamba Architectures

Mambular is a Python package that brings the power of advanced deep learning architectures to tabular data, offering a suite of models for regression, classification, and distributional regression tasks. Designed with ease of use in mind, Mambular models adhere to scikit-learn's `BaseEstimator` interface, making them highly compatible with the familiar scikit-learn ecosystem. This means you can fit, predict, and evaluate using Mambular models just as you would with any traditional scikit-learn model, but with the added performance and flexibility of deep learning.

## Features

- **Comprehensive Model Suite**: Includes modules for regression, classification, and distributional regression, catering to a wide range of tabular data tasks.
- **State-of-the-Art Architectures**: Leverages various advanced architectures known for their effectiveness in handling tabular data. Mambular models include powerful Mamba blocks [Gu and Dao](https://arxiv.org/pdf/2312.00752) and can include bidirectional processing as well as feature interaction layers.
- **Seamless Integration**: Designed to work effortlessly with scikit-learn, allowing for easy inclusion in existing machine learning pipelines, cross-validation, and hyperparameter tuning workflows.
- **Extensive Preprocessing**: Comes with a powerful preprocessing module that supports a broad array of data transformation techniques, ensuring that your data is optimally prepared for model training.
- **Sklearn-like API**: The familiar scikit-learn `fit`, `predict`, and `predict_proba` methods mean minimal learning curve for those already accustomed to scikit-learn.
- **PyTorch Lightning Under the Hood**: Built on top of PyTorch Lightning, Mambular models benefit from streamlined training processes, easy customization, and advanced features like distributed training and 16-bit precision.



## Models

| Model               | Description                                                                                      |
|---------------------|--------------------------------------------------------------------------------------------------|
| `Mambular`          | An advanced model using Mamba blocks [Gu and Dao](https://arxiv.org/pdf/2312.00752)  specifically designed for various tabular data tasks.       |
| `FTTransformer`     | A model leveraging transformer encoders, as introduced by [Gorishniy et al.](https://arxiv.org/abs/2106.11959), for tabular data. |
| `MLP`               | A classical Multi-Layer Perceptron (MLP) model for handling tabular data tasks.                  |
| `ResNet`            | An adaptation of the ResNet architecture for tabular data applications.                          |
| `TabTransformer`    | A transformer-based model for tabular data introduced by [Huang et al.](https://arxiv.org/abs/2012.06678), enhancing feature learning capabilities. |

All models are available for `regression`, `classification` and distributional regression, denoted by `LSS`.
Hence, they are available as e.g. `MambularRegressor`, `MambularClassifier` or `MambularLSS`



## Documentation

You can find the Mamba-Tabular API documentation [here](https://mambular.readthedocs.io/en/latest/).

## Installation

Install Mambular using pip:
```sh
pip install mambular
```

## Preprocessing

Mambular simplifies the preprocessing stage of model development with a comprehensive set of techniques to prepare your data for Mamba architectures. Our preprocessing module is designed to be both powerful and easy to use, offering a variety of options to efficiently transform your tabular data.

### Data Type Detection and Transformation

Mambular automatically identifies the type of each feature in your dataset and applies the most appropriate transformations for numerical and categorical variables. This includes:
- **Ordinal Encoding**: Categorical features are seamlessly transformed into numerical values, preserving their inherent order and making them model-ready.
- **One-Hot Encoding**: For nominal data, Mambular employs one-hot encoding to capture the presence or absence of categories without imposing ordinality.
- **Binning**: Numerical features can be discretized into bins, a useful technique for handling continuous variables in certain modeling contexts.
- **Decision Tree Binning**: Optionally, Mambular can use decision trees to find the optimal binning strategy for numerical features, enhancing model interpretability and performance.
- **Normalization**: Mambular can easily handle numerical features without specifically turning them into categorical features. Standard preprocessing steps such as normalization per feature are possible.
- **Standardization**: Similarly, standardization instead of normalization can be used to scale features based on the mean and standard deviation.
- **PLE (Periodic Linear Encoding)**: This technique can be applied to numerical features to enhance the performance of tabular deep learning methods by encoding periodicity.
- **Quantile Transformation**: Numerical features can be transformed to follow a uniform or normal distribution, improving model robustness to outliers.
- **Spline Transformation**: Applies piecewise polynomial functions to numerical features, capturing nonlinear relationships more effectively.
- **Polynomial Features**: Generates polynomial and interaction features, increasing the feature space to capture more complex relationships within the data.



### Handling Missing Values

Our preprocessing pipeline effectively handles missing data by using mean imputation for numerical features and mode imputation for categorical features. This ensures that your models receive complete data inputs without needing manual intervention.
Additionally, Mambular can manage unknown categorical values during inference by incorporating classical <UNK> tokens in categorical preprocessing.


## Fit a Model
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


## Distributional Regression with MambularLSS

Mambular introduces an approach to distributional regression through its `MambularLSS` module, allowing users to model the full distribution of a response variable, not just its mean. This method is particularly valuable in scenarios where understanding the variability, skewness, or kurtosis of the response distribution is as crucial as predicting its central tendency. All available moedls in mambular are also available as distributional models.

### Key Features of MambularLSS:

- **Full Distribution Modeling**: Unlike traditional regression models that predict a single value (e.g., the mean), `MambularLSS` models the entire distribution of the response variable. This allows for more informative predictions, including quantiles, variance, and higher moments.
- **Customizable Distribution Types**: `MambularLSS` supports a variety of distribution families (e.g., Gaussian, Poisson, Binomial), making it adaptable to different types of response variables, from continuous to count data.
- **Location, Scale, Shape Parameters**: The model predicts parameters corresponding to the location, scale, and shape of the distribution, offering a nuanced understanding of the data's underlying distributional characteristics.
- **Enhanced Predictive Uncertainty**: By modeling the full distribution, `MambularLSS` provides richer information on predictive uncertainty, enabling more robust decision-making processes in uncertain environments.



### Available Distribution Classes:

`MambularLSS` offers a wide range of distribution classes to cater to various statistical modeling needs. The available distribution classes include:

- `normal`: Normal Distribution for modeling continuous data with a symmetric distribution around the mean.
- `poisson`: Poisson Distribution for modeling count data that for instance represent the number of events occurring within a fixed interval.
- `gamma`: Gamma Distribution for modeling continuous data that is skewed and bounded at zero, often used for waiting times.
- `beta`: Beta Distribution for modeling data that is bounded between 0 and 1, useful for proportions and percentages.
- `dirichlet`: Dirichlet Distribution for modeling multivariate data where individual components are correlated, and the sum is constrained to 1.
- `studentt`: Student's T-Distribution for modeling data with heavier tails than the normal distribution, useful when the sample size is small.
- `negativebinom`: Negative Binomial Distribution for modeling count data with over-dispersion relative to the Poisson distribution.
- `inversegamma`: Inverse Gamma Distribution, often used as a prior distribution in Bayesian inference for scale parameters.
- `categorical`: Categorical Distribution for modeling categorical data with more than two categories.

These distribution classes allow `MambularLSS` to flexibly model a wide variety of data types and distributions, providing users with the tools needed to capture the full complexity of their data.


### Getting Started with MambularLSS:

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


### Implement Your Own Model

Mambular allows users to easily integrate their custom models into the existing logic. This process is designed to be straightforward, making it simple to create a PyTorch model and define its forward pass. Instead of inheriting from `nn.Module`, you inherit from Mambular's `BaseModel`. Each Mambular model takes three main arguments: the number of classes (e.g., 1 for regression or 2 for binary classification), `cat_feature_info`, and `num_feature_info` for categorical and numerical feature information, respectively. Additionally, you can provide a config argument, which can either be a custom configuration or one of the provided default configs.

One of the key advantages of using Mambular is that the inputs to the forward passes are lists of tensors. While this might be unconventional, it is highly beneficial for models that treat different data types differently. For example, the TabTransformer model leverages this feature to handle categorical and numerical data separately, applying different transformations and processing steps to each type of data.

Here's how you can implement a custom model with Mambular:


1. First, define your config:
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

2. Second, define your model:
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

3. Leverage the Mambular API:
You can build a regression, classification or distributional regression model that can leverage all of mambulars built-in methods, by using the following:

```python
from mambular.models import SklearnBaseRegressor

class MyRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MyCustomModel, config=MyConfig, **kwargs)
```

4. Train and evaluate your model:
You can now fit, evaluate, and predict with your custom model just like with any other Mambular model. For classification or distributional regression, inherit from `SklearnBaseClassifier` or `SklearnBaseLSS` respectively.

```python
regressor = MyRegressor(numerical_preprocessing="ple")
regressor.fit(X_train, y_train, max_epochs=50)
```

## Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@misc{2024,
    title={Mambular: Tabular Deep Learning with Mamba Architectures},
    author={Anton Frederik Thielmann, Manish Kumar, Christoph Weisser, Benjamin Saefken, Soheila Samiee},
    howpublished = {\url{https://github.com/basf/mamba-tabular}},
    year={2024}
}
```

## License

The entire codebase is under MIT license.
