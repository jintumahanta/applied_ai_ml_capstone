# Applied AI & ML Capstone Project

## Part 1 — Data Acquisition, Cleaning, and Exploratory Data Analysis

This repository contains Part 1 of the Applied AI & ML Capstone Project. The objective of this part is to inspect a raw tabular dataset, clean missing and duplicate data, correct data types, perform exploratory data analysis, study correlations and skewness, and prepare a cleaned dataset for machine learning tasks in Parts 2 and 3.

---

## Dataset Description and Justification

The Ames Iowa Housing dataset was selected for this project. The dataset contains residential property records with a combination of numerical and categorical variables describing property size, quality, construction year, neighborhood, garage characteristics, basement characteristics, and sale information.

The original training dataset contained 2,197 rows and 82 columns.

The dataset is suitable for the capstone requirements because it contains more than 500 observations, more than five numerical variables, multiple categorical variables, and a clearly identifiable target variable, `SalePrice`.

`SalePrice` is a continuous numerical target suitable for regression. It can also be converted into a binary target using a selected price threshold for a classification task in later parts of the project.

---

## 1. Data Loading and Initial Inspection

The dataset was loaded into a pandas DataFrame using `pd.read_csv()`.

The first five rows, column data types, and DataFrame shape were printed to understand the structure of the dataset.

Initial dataset shape:

- Rows: 2,197
- Columns: 82

The dataset contains both numerical and categorical features.

---

## 2. Null Value Analysis

The number and percentage of missing values were calculated for every column.

The following columns contained more than 20% missing values:

- `Alley`
- `Mas Vnr Type`
- `Fireplace Qu`
- `Pool QC`
- `Fence`
- `Misc Feature`

These columns were removed because the proportion of missing information was very high and imputing such a large percentage of values could introduce artificial patterns into the dataset.

For numerical columns with missing values below the 20% threshold, median imputation was used.

The median was preferred over the mean because it is less sensitive to extreme values and skewed distributions. Housing variables such as lot dimensions, basement area, and masonry veneer area can contain unusually large observations that may pull the mean away from the typical value.

Remaining categorical missing values were imputed using the mode of each categorical feature.

After the complete missing-value treatment, the final dataset contains zero missing values.

---

## 3. Duplicate Detection and Removal

Duplicate rows were identified using `df.duplicated().sum()`.

Number of duplicate rows detected: **0**

Rows before duplicate removal: **2,197**

Rows after duplicate removal: **2,197**

Rows removed: **0**

Since no duplicate rows were present, duplicate removal did not change the null percentage of any column.

---

## 4. Data Type Correction

`MS SubClass` was initially stored as an integer. However, the values represent dwelling classification codes rather than continuous numerical measurements. Therefore, the column was converted from `int64` to the `category` data type.

`Neighborhood` was initially stored as an object column and was converted to the `category` data type because it contains repeated categorical labels.

Memory usage before conversion:

**4,958,350 bytes**

Memory usage after conversion:

**4,826,507 bytes**

Memory saved:

**131,843 bytes**

Memory reduction:

**2.66%**

The conversion demonstrates that appropriate categorical data types can reduce DataFrame memory usage.

---

## 5. Descriptive Statistics and Skewness

Descriptive statistics were calculated for all numerical columns using `df.describe()`.

Skewness was calculated for each numerical feature.

The column with the highest absolute skewness was:

**Misc Val**

Skewness of `Misc Val`:

**20.1671**

The two numerical columns with the highest absolute skewness were:

1. `Misc Val` — 20.1671
2. `Pool Area` — 15.2828

Both variables are strongly positively skewed.

Positive skewness indicates that most observations are concentrated around lower values while a small number of observations contain extremely large values. These extreme high values create a long right tail in the distribution.

For positively skewed variables, the mean is pulled upward by extreme high observations. Therefore, the median provides a more representative measure of central tendency and is preferred for missing-value imputation.

---

## 6. Outlier Detection Using the IQR Method

The Interquartile Range method was applied to `SalePrice` and `Lot Area`.

### SalePrice

Q1: **130,000**

Q3: **215,000**

IQR: **85,000**

Lower bound: **2,500**

Upper bound: **342,500**

Number of outlier rows: **97**

### Lot Area

Q1: **7,500**

Q3: **11,660**

IQR: **4,160**

Lower bound: **1,260**

Upper bound: **17,900**

Number of outlier rows: **99**

The detected outliers were retained.

Large sale prices and unusually large lot areas may represent genuine properties rather than incorrect observations. Removing these records could eliminate valid information about premium or unusually large houses.

In Part 2, transformations or models that are less sensitive to extreme observations may be considered instead of directly deleting the outliers.

---

## 7. Exploratory Data Visualizations

Five visualization types were produced using Matplotlib and Seaborn.

### Line Plot — Sale Price Across Ordered Observations

The line plot shows substantial variation in sale prices across the ordered observations. Most houses fall within a moderate price range, while several sharp peaks represent high-value properties.

### Bar Chart — Mean Sale Price by Neighborhood

The bar chart compares the average sale price across neighborhoods.

A clear difference in average sale prices is visible between neighborhoods. This suggests that neighborhood location may contain useful predictive information for house prices.

### Histogram — Miscellaneous Value

The histogram of `Misc Val` shows an extremely right-skewed distribution.

Most observations are concentrated near zero, while a small number of very large values create a long right tail. This visual pattern is consistent with the calculated skewness value of 20.1671.

### Scatter Plot — Above-Ground Living Area vs Sale Price

The scatter plot between `Gr Liv Area` and `SalePrice` shows a strong positive relationship.

In general, houses with larger above-ground living areas tend to have higher sale prices. However, a few observations with very large living areas and comparatively low sale prices are visible.

These unusual observations may influence a linear regression model and should be considered during modelling.

### Box Plot — Sale Price by Overall Quality

The box plot shows that the median sale price generally increases as `Overall Qual` increases.

Higher quality ratings are associated with higher sale prices. The spread of sale prices also becomes larger for some high-quality categories, indicating greater price variability among premium properties.

---

## 8. Pearson Correlation Analysis

A Pearson correlation matrix was calculated for all numerical variables and visualized using an annotated heat map.

The pair with the highest absolute Pearson correlation was:

**Order and Yr Sold**

Pearson correlation:

**-0.9755**

This strong negative correlation should not be interpreted as a causal relationship.

`Order` is a record-order identifier and does not logically cause the year of sale to change. The relationship is more likely caused by the way observations were ordered or collected in the dataset.

Therefore, a third factor related to dataset construction or chronological record ordering provides a more plausible explanation for the observed correlation.

This result demonstrates that a high correlation value alone does not establish causation or guarantee that a feature is useful for predictive modelling.

---

## 9. Imputation Strategy Comparison

The two numerical variables with the highest absolute skewness were compared using their mean and median values.

### Misc Val

Mean: **57.4042**

Median: **0**

Skewness: **20.1671**

### Pool Area

Mean: **2.7897**

Median: **0**

Skewness: **15.2828**

Both variables have strong positive skewness.

For a positively skewed distribution, extreme high observations pull the mean upward. The median is less affected by these extreme observations and provides a more representative measure of the typical value.

Therefore, median imputation was selected for these variables.

After applying the selected imputation strategy, `isnull().sum()` confirmed that no missing values remained in either `Misc Val` or `Pool Area`.

---

## 10. Spearman Rank Correlation Analysis

A Spearman rank correlation matrix was calculated and compared with the Pearson correlation matrix.

The three column pairs with the largest absolute difference between Spearman and Pearson correlation were:

### Year Built and Open Porch SF

Absolute difference:

**0.2174**

The absolute Spearman correlation was greater than the absolute Pearson correlation. This suggests that the variables may follow a consistent monotonic relationship that is not completely linear.

### Garage Yr Blt and Open Porch SF

Absolute difference:

**0.1907**

Spearman correlation was stronger than Pearson correlation, suggesting a potentially monotonic but non-linear relationship.

### Lot Frontage and Lot Area

Absolute difference:

**0.1898**

Spearman correlation was again stronger than Pearson correlation. Lot frontage and lot area may generally increase together, but not in a strictly proportional linear pattern.

For these relationships, Spearman correlation will be considered for feature-selection guidance in Part 2 because rank-based correlation can capture monotonic relationships that Pearson correlation may underestimate.

---

## 11. Grouped Aggregation Analysis

`Neighborhood` was selected as the categorical variable and `SalePrice` as the numerical variable.

The grouped aggregation calculated the mean, standard deviation, and count of sale prices for every neighborhood.

The neighborhood with the highest mean sale price was:

**NoRidge**

Mean SalePrice:

**338,829.96**

The neighborhood with the highest standard deviation was:

**StoneBr**

Standard deviation:

**126,398.99**

The high within-group standard deviation indicates substantial sale-price variation among houses within the same neighborhood.

Therefore, neighborhood alone is insufficient to reliably predict the sale price of an individual property. Other variables such as overall quality, living area, construction year, and garage characteristics should also be considered.

The ratio of the highest neighborhood mean to the lowest neighborhood mean was:

**3.447**

This ratio is relatively large and indicates substantial differences in average sale prices across neighborhoods.

Therefore, `Neighborhood` appears to carry useful predictive signal for `SalePrice`, although it should be combined with other features in a predictive model.

---

## 12. Cleaned Dataset

The final cleaned dataset was saved as:

`cleaned_data.csv`

Final dataset shape:

**2,197 rows × 76 columns**

Total remaining missing values:

**0**

The cleaned dataset will be used for machine learning tasks in Parts 2 and 3.

---

## Repository Files

- `Part_1_EDA.ipynb` — Complete data cleaning and exploratory data analysis notebook.
- `cleaned_data.csv` — Final cleaned dataset with zero missing values.
- `README.md` — Documentation of methodology, design decisions, results, and interpretations.

---

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

---

## Conclusion

Part 1 established a complete data acquisition, cleaning, and exploratory analysis workflow for the Ames Iowa Housing dataset.

The analysis identified highly skewed variables, meaningful outliers, differences in housing prices across neighborhoods, and important relationships between numerical features.

The cleaned dataset contains 2,197 observations and 76 features with zero remaining missing values and is ready for supervised machine learning and feature-selection tasks in the next stages of the project.

---

# Part 2 — Supervised Machine Learning Model: Build, Train, and Evaluate

Part 2 develops and evaluates supervised machine learning models using the cleaned Ames Iowa Housing dataset produced in Part 1. Two predictive tasks were considered: regression for continuous house-price prediction and binary classification for distinguishing higher-priced houses from lower-priced houses.

## Target Variable Definitions

The regression target was defined as:

`y_reg = SalePrice`

SalePrice is a continuous numerical variable representing the selling price of a house.

For classification, a binary target was derived from the median SalePrice of the dataset. The median SalePrice was **163,500**.

The classification label was defined as:

- `1` — SalePrice is greater than the median SalePrice.
- `0` — SalePrice is less than or equal to the median SalePrice.

This produced a nearly balanced binary classification problem with 1,099 observations in class 0 and 1,098 observations in class 1.

## Feature Encoding

The feature matrix initially contained 37 categorical columns. Categorical variables with a meaningful natural order were encoded using ordinal mappings that preserved their ranking.

The following columns were ordinal encoded:

- Exter Qual
- Exter Cond
- Bsmt Qual
- Bsmt Cond
- Heating QC
- Kitchen Qual
- Garage Qual
- Garage Cond

For example, quality levels such as Poor, Fair, Typical, Good, and Excellent have an inherent ranking. Mapping these categories to ordered numerical values preserves this relationship.

The remaining 29 nominal categorical variables were one-hot encoded. One-hot encoding was used because categories such as neighborhood names or building types do not possess a meaningful numerical order. Assigning arbitrary integer labels to these variables could introduce a false ordinal relationship.

After encoding, the feature matrix contained **221 numerical features** and no remaining missing or non-numeric values.

## Leak-Free Train-Test Split and Feature Scaling

The dataset was divided into training and testing sets using an 80:20 train-test split with `random_state=42`.

The resulting split contained:

- 1,757 training observations.
- 440 testing observations.

A StandardScaler was fitted **only on the training features**. The fitted scaler was then used to transform both the training and testing features.

Fitting the scaler on the complete dataset would expose information about the distribution of the test data to the training process. This would create data leakage and could produce overly optimistic model evaluation results. Fitting preprocessing transformations only on the training data ensures that the test set remains unseen during model development.

## Linear Regression Model

A Linear Regression model was trained on the scaled training features.

The model achieved:

- Mean Squared Error (MSE): **964,387,322.16**
- R-squared (R²): **0.8179**

The R² value indicates that approximately **81.79% of the variation in SalePrice is explained by the regression model**.

The three features with the largest absolute Linear Regression coefficients were:

| Feature | Coefficient | Absolute Coefficient |
|---|---:|---:|
| BsmtFin SF 1 | 143,576.11 | 143,576.11 |
| Bsmt Unf SF | 126,048.61 | 126,048.61 |
| Total Bsmt SF | -124,004.69 | 124,004.69 |

Because the features were standardized, a large positive coefficient indicates that a one-standard-deviation increase in the feature is associated with a large increase in the predicted SalePrice while holding other features constant.

A large negative coefficient indicates that a one-standard-deviation increase in the feature is associated with a decrease in the predicted SalePrice while other model features remain constant.

The large and opposing coefficients among basement-area variables may also indicate strong relationships or multicollinearity between related basement measurements.

## Ridge Regression Comparison

Ridge Regression was trained using `alpha=1.0` with the same training and testing split.

The model comparison was:

| Model | MSE | R² |
|---|---:|---:|
| Linear Regression | 964,387,322.16 | 0.817884 |
| Ridge Regression | 963,372,158.92 | 0.818076 |

Ridge Regression produced a slightly lower MSE and a slightly higher R² than ordinary Linear Regression. The improvement was small, indicating broadly similar predictive performance.

Ridge Regression uses L2 regularization, which adds a penalty based on the squared magnitude of model coefficients. The `alpha` parameter controls the strength of this penalty. A larger alpha applies stronger regularization and shrinks coefficients more aggressively.

Ridge Regression may produce a different coefficient profile from ordinary least squares because correlated predictors can cause unstable and very large OLS coefficients. The L2 penalty reduces coefficient magnitude and distributes predictive influence more smoothly across correlated features.

The three largest absolute Ridge coefficients were associated with `Roof Matl_CompShg`, `Roof Matl_Tar&Grv`, and `Roof Matl_WdShngl`.

## Logistic Regression Classification Model

The classification training set contained:

- Class 0: **879 observations**
- Class 1: **878 observations**

The minority class represented approximately **49.97%** of the training data. Therefore, the classes were sufficiently balanced and neither SMOTE nor `class_weight='balanced'` was required.

A Logistic Regression model with `max_iter=1000` was trained using the scaled training features.

The confusion matrix was:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 206 | 14 |
| Actual 1 | 9 | 211 |

The classification model achieved approximately:

- Accuracy: **0.95**
- Precision for class 1: **0.94**
- Recall for class 1: **0.96**
- F1-score for class 1: **0.95**
- ROC-AUC: **0.97598**

These results indicate strong classification performance.

## Precision and Recall

Precision is calculated as:

`Precision = TP / (TP + FP)`

Precision measures the proportion of predicted positive observations that are actually positive.

Recall is calculated as:

`Recall = TP / (TP + FN)`

Recall measures the proportion of actual positive observations that are correctly identified by the model.

For this classification task, class 1 represents houses with a SalePrice greater than the dataset median. Precision is important when incorrectly identifying a lower-priced house as a higher-priced house is costly. Recall is important when failing to identify genuinely higher-priced houses is more costly.

For a housing price screening task, recall can be considered particularly important because missing genuinely higher-priced properties may result in missed investment, valuation, or sales opportunities.

## ROC Curve and AUC Interpretation

The Logistic Regression model achieved a ROC-AUC score of **0.97598**.

The ROC curve evaluates the relationship between the True Positive Rate and False Positive Rate across multiple classification thresholds.

An AUC value close to 1 indicates that the model has a strong ability to distinguish between the two classes. Therefore, an AUC of approximately 0.976 indicates excellent discrimination between houses above and below the median SalePrice.

## Decision-Threshold Sensitivity Analysis

Predicted probabilities were evaluated using decision thresholds from 0.30 to 0.70.

| Threshold | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| 0.30 | 0.918103 | 0.968182 | 0.942478 |
| 0.40 | 0.929515 | 0.959091 | 0.944072 |
| 0.50 | 0.937778 | 0.959091 | 0.948315 |
| 0.60 | 0.940639 | 0.936364 | 0.938497 |
| 0.70 | 0.947867 | 0.909091 | 0.928074 |

The threshold with the maximum F1 score was **0.50**, with an F1 score of approximately **0.9483**.

Increasing the classification threshold generally increased precision but reduced recall. Lowering the threshold increased recall because more observations were classified as positive, although this also increased the possibility of false positives.

For this housing classification problem, recall is considered important because false negatives correspond to genuinely higher-priced houses incorrectly classified as lower-priced. To optimize recall, the decision threshold could be lowered below 0.50. However, lowering the threshold would increase false positives and reduce precision.

The default threshold of **0.50** provides the highest F1 score among the evaluated thresholds and therefore provides the best observed balance between precision and recall.

## Logistic Regression Regularization Experiment

A second Logistic Regression model was trained using `C=0.01`, representing stronger L2 regularization.

The comparison was:

| Model | Precision | Recall | AUC |
|---|---:|---:|---:|
| Baseline Logistic Regression (C=1.0) | 0.937778 | 0.959091 | 0.975981 |
| Strong Regularization (C=0.01) | 0.951220 | 0.886364 | 0.978306 |

Reducing C from 1.0 to 0.01 increased precision by approximately **0.01344** and increased AUC by approximately **0.00232**. However, recall decreased by approximately **0.07273**.

In Logistic Regression, the parameter C is the inverse of regularization strength. A smaller C applies stronger L2 regularization and shrinks model coefficients more aggressively.

On this dataset, stronger regularization slightly improved precision and AUC but substantially reduced recall. Therefore, the strongly regularized model was less effective at identifying all genuinely higher-priced houses.

## Bootstrap Confidence Interval for AUC Difference

A bootstrap analysis using **500 samples drawn with replacement** was performed to evaluate the reliability of the AUC difference between the baseline Logistic Regression model (`C=1.0`) and the strongly regularized model (`C=0.01`).

The AUC difference was defined as:

`AUC Difference = AUC(C=1.0) - AUC(C=0.01)`

The bootstrap results were:

- Mean AUC difference: **-0.00245**
- 2.5th percentile: **-0.00966**
- 97.5th percentile: **0.00386**
- 95% bootstrap confidence interval: **[-0.00966, 0.00386]**

The 95% confidence interval includes zero. Therefore, the observed AUC difference between the two Logistic Regression models may not be statistically reliable across different samples.

Although the `C=0.01` model produced a slightly higher AUC on the current test set, the bootstrap analysis indicates that its advantage is not consistently maintained across resampled datasets.

## Part 2 Conclusion

Part 2 successfully developed regression and classification models using the cleaned Ames Iowa Housing dataset.

Linear Regression explained approximately **81.79% of the variance in SalePrice**, while Ridge Regression produced a small improvement in MSE and R². Logistic Regression achieved approximately **95% classification accuracy and a ROC-AUC of 0.976**, demonstrating strong separation between houses above and below the median SalePrice.

Decision-threshold analysis showed that the default threshold of 0.50 produced the highest F1 score among the tested thresholds. Stronger Logistic Regression regularization slightly improved precision and AUC but reduced recall. Finally, bootstrap analysis showed that the small AUC difference between the two Logistic Regression models was not reliably different from zero.

The models and evaluation workflow developed in this part provide a foundation for further machine learning analysis and model development.

---

# Part 3 — Advanced Modeling, Ensembles, Tuning, and Full ML Pipeline

## Overview

Part 3 extends the supervised machine learning workflow developed in Part 2 by evaluating advanced classification models, ensemble learning techniques, hyperparameter tuning, feature importance, feature ablation, cross-validation, and model serialization.

The objective of this stage is to identify the most robust classification model for predicting whether a house belongs to the higher-price category.

The classification target remains:

- `1` — SalePrice greater than the median SalePrice.
- `0` — SalePrice less than or equal to the median SalePrice.

The same leak-free train-test split and preprocessing workflow from Part 2 were reused.

---

## Decision Tree Baseline

An unconstrained `DecisionTreeClassifier` was trained using the scaled classification training data.

The results were:

| Metric | Value |
|---|---:|
| Training Accuracy | 1.0000 |
| Test Accuracy | 0.8773 |
| Train-Test Accuracy Gap | 0.1227 |

The model achieved perfect training accuracy but substantially lower test accuracy. The train-test accuracy gap of approximately `0.123` indicates clear signs of overfitting.

An unconstrained decision tree can grow until the training observations are separated almost perfectly. Decision trees are high-variance models because they greedily select the locally best split at each node without revisiting earlier decisions. Consequently, the model may learn noise and dataset-specific patterns that do not generalize well to unseen observations.

---

## Controlled Decision Tree

A second Decision Tree model was trained using the following constraints:

- `max_depth = 5`
- `min_samples_split = 20`

The results were:

| Metric | Unconstrained Tree | Controlled Tree |
|---|---:|---:|
| Training Accuracy | 1.0000 | 0.9294 |
| Test Accuracy | 0.8773 | 0.8977 |
| Train-Test Accuracy Gap | 0.1227 | 0.0317 |

The controlled Decision Tree substantially reduced the train-test accuracy gap from approximately `0.123` to `0.032`.

The `max_depth` parameter limits how deeply the tree can grow. Limiting tree depth reduces model variance but may introduce additional bias.

The `min_samples_split` parameter prevents a node from being split when fewer than the specified number of samples are present. This reduces the likelihood of creating highly specific splits based on noise in small subsets of the training data.

The controlled tree therefore demonstrated better generalization than the unconstrained Decision Tree.

---

## Gini Impurity and Entropy Comparison

Two controlled Decision Tree classifiers with `max_depth=5` were compared using the Gini and Entropy splitting criteria.

The test accuracies were:

| Criterion | Test Accuracy |
|---|---:|
| Gini | 0.8909 |
| Entropy | 0.8909 |

Both criteria produced identical test accuracy on this dataset.

### Gini Impurity

The Gini impurity is defined as:

`Gini = 1 - Σ(p_i²)`

where `p_i` represents the proportion of samples belonging to class `i`.

A Gini impurity of `0` indicates that all observations in a node belong to the same class.

### Entropy

Entropy is defined as:

`Entropy = -Σ(p_i log₂(p_i))`

Entropy measures the uncertainty or disorder in a node. An entropy value of `0` indicates a completely pure node where all observations belong to one class.

The identical test accuracy suggests that the choice between Gini impurity and Entropy had little practical impact on the predictive performance of the controlled Decision Tree for this dataset.

---

## Random Forest Model

A `RandomForestClassifier` was trained using:

- `n_estimators = 100`
- `max_depth = 10`
- `random_state = 42`

The results were:

| Metric | Value |
|---|---:|
| Training Accuracy | 0.9943 |
| Test Accuracy | 0.9227 |
| ROC-AUC | 0.9817 |

The Random Forest substantially improved test performance compared with the individual Decision Tree models.

### Bagging Concept

Random Forest uses bagging, or bootstrap aggregating. Each tree is trained on a bootstrap sample drawn with replacement from the training dataset. In addition, each split evaluates only a random subset of approximately `√(number of features)` features.

Because the individual trees observe different samples and different subsets of features, their prediction errors are less correlated. The Random Forest combines the predictions of multiple trees, reducing variance and producing more stable predictions than a single deep Decision Tree.

---

## Random Forest Feature Importance

The five most important features identified by the Random Forest were:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | Overall Qual | 0.100206 |
| 2 | Gr Liv Area | 0.059564 |
| 3 | Year Built | 0.053583 |
| 4 | Garage Yr Blt | 0.048437 |
| 5 | Exter Qual | 0.045618 |

Random Forest feature importance represents the average reduction in Gini impurity produced by splits involving a feature across all trees in the ensemble.

This differs from a Linear Regression coefficient. A regression coefficient represents the expected change in the predicted target associated with a one-unit change in a feature while holding other variables constant. Random Forest feature importance does not represent the direction or magnitude of a direct linear effect. Instead, it measures how useful a feature is for reducing classification impurity throughout the ensemble.

The results indicate that overall construction quality, above-ground living area, building age, garage age, and exterior quality were particularly influential when classifying houses into higher- and lower-price groups.

---

## Gradient Boosting Model

A `GradientBoostingClassifier` was trained using:

- `n_estimators = 100`
- `learning_rate = 0.1`
- `max_depth = 3`
- `random_state = 42`

The results were:

| Metric | Value |
|---|---:|
| Training Accuracy | 0.9841 |
| Test Accuracy | 0.9341 |
| ROC-AUC | 0.9798 |

Gradient Boosting achieved the highest test accuracy among the evaluated models.

Unlike Random Forest, where trees are trained largely independently, Gradient Boosting builds trees sequentially. Each new tree attempts to correct errors made by the previous ensemble.

Although Gradient Boosting achieved slightly higher test accuracy, its ROC-AUC was slightly lower than the Random Forest ROC-AUC.

---

## Feature Ablation Study

The five features with the lowest Random Forest feature importance were:

1. `Electrical_Mix`
2. `Sale Type_ConLw`
3. `Sale Type_ConLI`
4. `Sale Type_Con`
5. `Sale Condition_AdjLand`

A second Random Forest classifier was trained after removing these five features.

The comparison was:

| Model | Feature Count | ROC-AUC |
|---|---:|---:|
| Full Random Forest | 221 | 0.981674 |
| Reduced Random Forest | 216 | 0.980331 |

The ROC-AUC difference between the reduced and full models was:

`0.980331 - 0.981674 = -0.001343`

Removing the five lowest-importance features caused a very small decrease in ROC-AUC.

This indicates that the removed features contained limited predictive information, although they were not completely uninformative. The small performance reduction suggests that removing them may be acceptable when a lower-dimensional model is desirable.

In a production environment, reducing the number of features may lower data collection requirements, inference complexity, and maintenance burden. However, feature removal is appropriate only when the resulting performance degradation remains below an acceptable threshold.

---

## Five-Fold Cross-Validation Comparison

A five-fold stratified cross-validation procedure was performed using ROC-AUC as the scoring metric.

The results were:

| Model | Mean CV ROC-AUC | CV ROC-AUC Standard Deviation |
|---|---:|---:|
| Controlled Decision Tree | 0.938457 | 0.014527 |
| Random Forest | 0.981130 | 0.006510 |
| Gradient Boosting | 0.981041 | 0.005779 |

Cross-validation provides a more reliable estimate of generalization performance than a single train-test split because the model is evaluated on multiple different validation subsets.

The Random Forest achieved the highest mean cross-validated ROC-AUC of approximately `0.98113`. Gradient Boosting achieved a nearly identical mean ROC-AUC and slightly lower standard deviation.

The Controlled Decision Tree showed both lower mean ROC-AUC and greater variation across folds.

---

## Hyperparameter Tuning with GridSearchCV

A machine learning pipeline was constructed using:

- `SimpleImputer(strategy='median')`
- `StandardScaler()`
- `RandomForestClassifier(random_state=42)`

The Random Forest hyperparameters evaluated were:

- `n_estimators`: 50, 100, 200
- `max_depth`: 5, 10, None
- `min_samples_leaf`: 1, 5

The complete parameter grid contained:

`3 × 3 × 2 = 18 model configurations`

All 18 configurations were evaluated using five-fold stratified cross-validation and ROC-AUC scoring.

The best parameters were:

- `max_depth = 10`
- `min_samples_leaf = 1`
- `n_estimators = 200`

The best cross-validated ROC-AUC score was:

`0.981810`

Grid Search was exhaustive because every possible combination in the defined parameter grid was evaluated.

In contrast, Randomized Search evaluates only a selected number of randomly sampled parameter combinations. Randomized Search can therefore be computationally more efficient for large hyperparameter spaces, while Grid Search is suitable for relatively small and carefully defined search spaces such as the one used in this project.

---

## Manual Learning Curve Analysis

The tuned pipeline was trained using progressively larger fractions of the training dataset.

The results were:

| Training Fraction | Training AUC | Test AUC |
|---|---:|---:|
| 0.20 | 1.000000 | 0.976250 |
| 0.40 | 1.000000 | 0.981715 |
| 0.60 | 0.999946 | 0.981798 |
| 0.80 | 0.999901 | 0.984835 |
| 1.00 | 0.999648 | 0.982314 |

The training AUC remained extremely high as the training dataset increased. It decreased slightly from `1.0000` to approximately `0.99965`, which is expected as a model encounters more diverse training observations.

The test AUC generally improved as the amount of training data increased, rising from approximately `0.9763` at the 20% training fraction to values around `0.982–0.985` for the larger training subsets.

The test AUC reached `0.984835` at the 80% training fraction and then decreased slightly to `0.982314` when the complete training dataset was used. This small fluctuation is consistent with normal sampling variation.

Overall, the test AUC appears to have largely plateaued around `0.98`. Therefore, the current model appears more capacity-limited than strongly data-limited. Additional training data may provide incremental improvements, but major performance gains would more likely require improved feature engineering or a different modeling approach.

---

## Best Model Serialization

The tuned Random Forest pipeline was saved using `joblib` as:

`best_model.pkl`

The saved model file size was approximately:

`3,951,874 bytes`

The saved model was successfully reloaded and used to generate predictions for two sample observations.

Predicted classes:

`[1, 0]`

Predicted probabilities for the higher-price class:

`[0.96665595, 0.05929494]`

The successful reload-and-predict test confirms that the serialized pipeline can perform predictions without retraining.

A saved model can be loaded and used as follows:

    import joblib

    loaded_model = joblib.load('best_model.pkl')

    sample_data = X_test_clf.iloc[:2]

    predictions = loaded_model.predict(sample_data)
    probabilities = loaded_model.predict_proba(sample_data)

    print(predictions)
    print(probabilities)

---

## Final Model Comparison

The final comparison of the classification models was:

| Model | 5-Fold CV Mean AUC | 5-Fold CV AUC Std | Test AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.974346 | 0.009301 | 0.975981 |
| Controlled Decision Tree | 0.938457 | 0.014527 | 0.940671 |
| Random Forest | 0.981130 | 0.006510 | 0.981674 |
| Gradient Boosting | 0.981041 | 0.005779 | 0.979814 |

---

## Final Model Recommendation

The **Random Forest model is recommended as the final production model**.

It achieved:

- Mean five-fold cross-validation ROC-AUC: `0.98113`
- Cross-validation ROC-AUC standard deviation: `0.00651`
- Test ROC-AUC: `0.98167`

Random Forest achieved the highest mean cross-validated ROC-AUC among the evaluated models and produced the highest test ROC-AUC.

Gradient Boosting produced slightly higher classification accuracy and a slightly lower cross-validation standard deviation. However, Random Forest demonstrated marginally better ROC-AUC performance across both cross-validation and the independent test set.

The feature ablation study also demonstrated that the Random Forest remained highly robust after removing its five lowest-importance features, with an ROC-AUC reduction of only approximately `0.00134`.

The learning curve showed that model performance had largely stabilized around a test ROC-AUC of `0.98`, suggesting strong generalization performance.

Therefore, the tuned Random Forest pipeline provides the best overall balance of predictive discrimination, cross-validation stability, model robustness, and deployment readiness for this classification problem.

---

## Part 3 Repository Files

- `Part_3_Advanced_Modeling.ipynb` — Advanced modeling, ensemble learning, feature ablation, cross-validation, hyperparameter tuning, and model serialization notebook.
- `best_model.pkl` — Serialized tuned Random Forest machine learning pipeline.
- `Part_2_Supervised_ML.ipynb` — Supervised machine learning notebook from Part 2.
- `Part_1_EDA.ipynb` — Data cleaning and exploratory data analysis notebook.
- `cleaned_data.csv` — Cleaned dataset used for machine learning.
- `README.md` — Complete project methodology, results, and interpretation.

---

## Technologies Used in Part 3

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- Jupyter Notebook / Google Colab

---

## Conclusion

Part 3 extended the project from baseline supervised learning to advanced ensemble modeling and a reproducible machine learning pipeline.

The analysis demonstrated the overfitting behavior of an unconstrained Decision Tree and showed how tree constraints substantially improved generalization. Random Forest and Gradient Boosting significantly outperformed the individual Decision Tree models.

Five-fold cross-validation showed that Random Forest and Gradient Boosting achieved mean ROC-AUC values of approximately `0.981`. GridSearchCV identified an optimized Random Forest configuration with 200 estimators, a maximum depth of 10, and a minimum leaf size of 1.

Feature ablation showed that removing the five least-important features caused only a minor ROC-AUC reduction. The manual learning curve indicated that model performance had largely stabilized near a test ROC-AUC of `0.98`.

The tuned Random Forest pipeline was serialized successfully and verified using a reload-and-predict test.

Based on cross-validation performance, test ROC-AUC, robustness, and deployment readiness, the Random Forest pipeline was selected as the final recommended model.
