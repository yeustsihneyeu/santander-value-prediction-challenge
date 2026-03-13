# Exploratory Data Analysis (EDA) Report

## 1. Dataset Overview

The dataset contains:

- **4459 observations (rows)**
- **4993 columns**, including the target variable
- **~4992 potential features**

Thus, the number of features exceeds the number of observations:

p > n

This configuration increases the **risk of overfitting**, especially for complex models.  
Therefore, **feature filtering, regularization, or dimensionality reduction** may be necessary during modeling.

Additional observations:

- Most features are **anonymous**, meaning their business interpretation is unknown.
- The dataset consists primarily of **numerical variables**.
- One column (`ID`) appears to be an identifier and should **not be used as a predictive feature**.

---

## 2. Target Variable Analysis

The target variable is **numeric and strictly positive**.

### Missing Values

No missing values were detected in the target column.

### Distribution

The distribution of the target variable shows **strong right skewness**:

- The **mean is significantly higher than the median**.
- The value range is large:

30,000 → 40,000,000

This indicates the presence of **large values and potential outliers**.

### Outlier Detection

Using the **IQR rule**, approximately **10.25% of observations** are detected as upper outliers.

However, this result should be interpreted carefully.  
These values may represent **valid high-magnitude observations rather than noise**, therefore removal should not be performed automatically.

### Log Transformation

A **log transformation** was tested:

log1p(target)

After transformation:

- the distribution becomes significantly **less skewed**
- variance is reduced
- the distribution becomes more **symmetric**

Because of this, it is reasonable to **evaluate both versions of the target** during modeling:

- original target
- log-transformed target

The final decision should be made using experiments.

---

## 3. Feature Quality Analysis

### Missing Values

No missing values were detected in the training dataset.

### Duplicate Observations

One duplicated row was identified.  
Before removal, it is recommended to verify that the **target value is identical**, to avoid accidental data leakage or inconsistencies.

### Duplicate Features

Approximately **260 duplicated feature columns** were detected.

Duplicated features provide **identical information** and may introduce:

- unnecessary dimensionality
- model instability
- increased computational cost

Therefore, these columns can safely be **removed during preprocessing**.

### Constant Features

At least **one constant feature** was detected.

Features with a single unique value contain **no predictive information**, so they should also be removed.

---

## 4. Sparsity Analysis

The dataset shows **strong sparsity**.

Many features contain a very high proportion of **zero values**.

To quantify sparsity, feature filtering was tested using thresholds:

- **90% zeros**
- **95% zeros**
- **99% zeros**

Results indicate that **around half of the features contain more than 99% zero values**.

Approximately **~2100 features** meet this condition.

These features may contain **very little signal** and can increase model noise.  
However, rare features can sometimes still be informative, particularly in tree-based models.

Therefore:

- removing these features can be tested as a **preprocessing experiment**
- the final decision should be validated using model performance

After applying this filtering rule, the feature space is reduced by **approximately half**, which significantly decreases model complexity.

---

## 5. Feature–Target Relationship

To investigate relationships between features and the target variable, three statistical methods were applied.

### Pearson Correlation

Pearson correlation measures **linear relationships**.

Results show that most features have **very low correlation values**, indicating weak linear relationships.

This suggests that **linear models may struggle** to capture the signal.

---

### Spearman Correlation

Spearman correlation measures **monotonic relationships**.

Results are similarly weak, indicating that most features do not have strong monotonic relationships with the target variable.

---

### Mutual Information

Mutual Information (MI) was used to detect **general statistical dependency**, including **non-linear relationships**.

The results indicate that approximately **~70 features** show noticeable mutual information with the target.

These features represent **potentially informative predictors** and should be investigated further.

Possible follow-up analysis includes:

- examining feature distributions (excluding zero values)
- checking for outliers
- testing transformations such as **log scaling**

It is important to note that the MI threshold used here is **heuristic**, and these features should be treated as **candidate features rather than final selections**.

---

## 6. Feature–Feature Relationships

Feature redundancy was analyzed using a **correlation matrix**.

The analysis revealed approximately **~30 feature pairs** with **very high correlation (> 0.9)**.

Highly correlated features may represent **redundant information**.

Removing one feature from such pairs can:

- reduce multicollinearity
- simplify the model
- improve training stability

However, the usefulness of this step depends on the **model type**:

- linear models benefit strongly
- tree-based models are less sensitive to multicollinearity

Therefore, feature removal can be tested as an **optional preprocessing step**.

---

## 7. Main Findings

The dataset exhibits several important characteristics.

### High Dimensionality

The number of features exceeds the number of observations:

p > n

This increases the risk of **overfitting**, making feature filtering important.

### Target Distribution

The target variable is **strongly right-skewed**, and a **log transformation** significantly improves the distribution.

Both versions of the target should be evaluated during modeling.

### Data Sparsity

A large portion of features contain **mostly zero values**, indicating strong sparsity.

Filtering features with **>99% zeros** reduces dimensionality significantly and should be evaluated experimentally.

### Weak Linear Relationships

Simple correlation analysis shows **weak linear and monotonic relationships** between individual features and the target.

However, **mutual information suggests the presence of non-linear signals** in a subset of features.

---

## 8. Modeling Implications

Based on the EDA results, several modeling strategies appear promising.

**Tree-based models**

- Random Forest
- Gradient Boosting
- LightGBM
- XGBoost

These models handle:

- sparse features
- nonlinear relationships
- high-dimensional datasets

relatively well.

---

## 9. Recommended Next Steps

1. Remove:
   - duplicated columns
   - constant features
   - features with >99% zeros (experimentally)

2. Train **baseline models** using:

- original target
- log-transformed target

3. Evaluate models using **cross-validation**.

4. Compare performance across:

- linear models (with regularization)
- tree-based models

5. Investigate **top features selected by Mutual Information** and model feature importance.

6. As an additional experiment, evaluate **dimensionality reduction techniques**, particularly **Principal Component Analysis (PCA)**.

The goal of this experiment is to:

- reduce the dimensionality of the feature space
- capture the most informative variance in a smaller set of components
- potentially improve model stability and training efficiency
