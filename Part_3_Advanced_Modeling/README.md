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
