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
