---
title: "Mambular: A User-Centric Python Library for Tabular Deep Learning Leveraging Mamba Architecture"
tags:
  - Python
  - Tabular Deep Learning
  - Mamba
  - Distributional Regression
authors:
  - name: Anton Thielmann
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Soheila Samiee
    affiliation: 2
  - name: Christoph Weisser
    affiliation: 1
affiliations:
  - name: BASF SE, Germany
    index: 1
  - name: BASF Canada Inc, Canada
    index: 2
date: 22 April 2024
bibliography: paper.bib

---



# 1. Summary

Mambular is a Python library designed to leverage the capabilities of the recently proposed Mamba architecture [@Gu] for deep learning tasks involving tabular datasets. The effectiveness of the attention mechanism, as demonstrated by models such as TabTransformer [@Ahamed] and FT-Transformer [@Gorishnyi1], is extended to these data types, showcasing the potential for sequence-focused architectures to excel in this domain. Thus, sequence-focused architectures can also achieve state-of-the-art performances for tabular data problems. [@Huang] already demonstrated that the Mamba architecture, similar to the attention mechanism, can effectively be used when dealing with tabular data. Mambular closely follows [@Gorishnyi1], but uses Mamba blocks instead of transformer blocks.
Furthermore, it offers enhanced flexibility in model architecture with respect to embedding activation, pooling layers, and task-specific head architectures. Choosing the appropriate settings, a user can thus easily implement the models presented in [@Huang].

# 2. Statement of Need
Transformer-based models for tabular data have become powerful alternatives to traditional gradient-based decision trees. [@Huang; @Gorishnyi1; @natt]. However, effectively training these models requires users to:  **i)** deeply understand the intricacies of tabular transformer networks,  **ii)** master various data type-dependent preprocessing techniques, **iii)** navigate complex deep learning libraries.
This either leads researchers and practitioners alike to develop extensive custom scripts and libraries to fit these models or discourages them from using these advanced tools altogether. However, since tabular transformer models are becoming more popular and powerful, they should be easy to use, also for practitioners. Mambular addresses this by offering a straightforward framework that allows users to easily train tabular models using the innovative Mamba architecture.

# 3. Methodology
The Mambular default architecture, independent of the task follows the straight forward architecture of tabular tansformer models [@Ahamed; @Gorishnyi1; @Huang]:
If the numerical features are integer binned they are treated as categorical features and each feature/variable is passed through an embedding layer. When other numerical preprocessing techniques are applied (or no preprocessing), the numerical features are passed through a single feed-forward dense layer with the same output dimensionality as the embedding layers [@Gorishnyi1]. By default, no activation is used on the created embeddings, but the users can easily change that with available arguments. The created embeddings are passed through a stack of Mamba layers after which the contextualized embeddings are pooled (default is average pooling). Mambular also offers the use of cls token embeddings instead of pooling layers. After pooling, RMS layer normalization from [@Gu] is applied by default, followed by a task-specific model head.

### 3.1 Models
Mambular includes the following three model classes:
**i)** *MambularRegressor* for regression tasks, **ii)** *MambularClassifier* for classification tasks and **iii)** *MambularLSS* for distributional regression tasks, similar to [@Thielmann].^[ See e.g. [@Kneib] for an overview on distributional regression.]


The loss functions are respectively the **i)** Mean squared error loss, **ii)** categorical cross entropy (Binary for binary classification) and **iii)** the negative log-likelihood for distributional regression. For **iii)** all distributional parameters have default activation/link functions that adhere to the distributional restrictions (e.g. positive variance for a normal distribution) but can be adapted to the users preferences. The inclusion of a distributional model focusing on regression beyond the mean further allows users to account for aleatoric uncertainty [@Kneib] without increasing the number of parameters or the complexity of the model.

# 4. Ecosystem Compatibility and Flexibility

Mambular is seamlessly compatible with the scikit-learn [@Pedregosa] ecosystem, allowing users to incorporate Mambular models into their existing workflows with minimal friction. This compatibility extends to various stages of the machine learning process, including data preprocessing, model training, evaluation, and hyperparameter tuning.

Furthermore, Mambular's design emphasizes flexibility and user-friendliness. The library offers a range of customizable options for model architecture, including the choice of preprocessing, activation functions, pooling layers, normalization layers, regularization and more. This level of customization ensures that practitioners can tailor their models to the specific requirements of their tabular data tasks, optimizing performance and achieving state-of-the-art results as demonstrated by [@Ahamed].



### 4.1 Preprocessing Capabilities

Mambular includes a comprehensive preprocessing module also following scikit-learns preprocessing pipeline. 
The preprocessing module supports a wide range of data transformation techniques, including ordinal and one-hot encoding for categorical variables, decision tree-based binning for numerical features, and various strategies for handling missing values. By leveraging these preprocessing tools, users can ensure that their data is in the best possible shape for training Mambular models, leading to improved model performance.

# Acknowledgements
We sincerely acknowledge and appreciate the financial support provided by the Key Digital Capability (KDC) for Generative AI at BASF and the BASF Data & AI Academy, which played a critical role in facilitating this research.

# References



