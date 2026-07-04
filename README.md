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
