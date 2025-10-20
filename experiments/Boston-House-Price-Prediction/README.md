# Boston House Price Prediction

This project reconstructs the exploratory analysis and linear regression modeling workflow for the classic Boston housing dataset. It highlights the predictors that carry the strongest signal for median home value while providing reproducible plots and diagnostics.

## Getting Started

- Use Python 3.12+ and create a virtual environment with uv: `uv venv`
- Activate it (`source .venv/bin/activate` on Linux)
- Install the project and dependencies: `uv pip install -e .`
- Run the analysis from the project root: `python main.py`

The script prints dataset diagnostics, fits the regression models, and writes every chart to the `result/` directory so they can be reviewed without opening the notebook.

## Insights Highlighted

- Lower crime, proximity to the Charles River, and access to urban amenities correlate with higher prices
- Industrial and older neighborhoods show distinct behavior that influences model selection
- Residual diagnostics and cross-validation keep the model assumptions transparent

---

# **Boston House Price Prediction - Linear Regression**

-------------------------------
## **Objective**
-------------------------------

The the goal of this project is to **predict the housing prices of a town or a suburb based on the features of the locality provided to us**. In the process, we need to **identify the most important features affecting the price of the house**. We need to employ techniques of data preprocessing and build a linear regression model that predicts the prices for the unseen data.

----------------------------
## **Dataset**
---------------------------

Each record in the database describes a house in Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. Detailed attribute information can be found below:

Attribute Information:

- **CRIM:** Per capita crime rate by town
- **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS:** Proportion of non-retail business acres per town
- **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- **NOX:** Nitric Oxide concentration (parts per 10 million)
- **RM:** The average number of rooms per dwelling
- **AGE:** Proportion of owner-occupied units built before 1940
- **DIS:** Weighted distances to five Boston employment centers
- **RAD:** Index of accessibility to radial highways
- **TAX:** Full-value property-tax rate per 10,000 dollars
- **PTRATIO:** Pupil-teacher ratio by town
- **LSTAT:** % lower status of the population
- **MEDV:** Median value of owner-occupied homes in 1000 dollars

## **Importing the necessary libraries and overview of the dataset**

### **Loading the data**

**Observation:**

* The price of the house indicated by the variable MEDV is the target variable and the rest of the variables are independent variables based on which we will predict the house price (MEDV).

### **Checking the info of the data**

**Observations:**

- There are a total of **506 non-null observations in each of the columns**. This indicates that there are **no missing values** in the data.
- There are **13 columns** in the dataset and **every column is of numeric data type**.

## **Exploratory Data Analysis and Data Preprocessing**

### **Summary Statistics of this Dataset**

**Observations:**
* **CRIM:** Per capita crime rate by town
    * Around 75% of the crime rate falls between ~0-4 with a max of 88 suggesting a possible **outlier** 
* **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.
    * Over 50% have 0% have residential land zoned for lots over 25,00sq.ft with the max 100%, suggesting this is **perhaps a rare commodity**.
* **INDUS:** Proportion of non-retail business acres per town
    * Ranges from 0.4-27% with an average of 11%, suggesting most towns have some industrial businesses.
* **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    * With a mean of 0.069 only ~7% of houses bound the Charles River.
* **NOX:** Nitric Oxide concentration (parts per 10 million)
    * Ranges from 0.38-0.87 with a average of 0.55. Distribution looks nominal.
* **RM:** The average number of rooms per dwelling
    * Ranges from 3.5-8.7 with an average of 6.2. Distribution looks nominal.
* **AGE:** Proportion of owner-occupied units built before 1940
    * Ranges from 2.9-100y with an averaga of 68y. Distribution looks nominal.
    * **Min age of 2.9y indicates that no houses in the database are newly built**
* **DIS:** Weighted distances to five Boston employment centers
    * Ranges form 1.1-12.1 with an average of 3.7. Distribution looks nominal.
* **RAD:** Index of accessibility to radial highways
    * Ranges from 1-24 with over 75% being the max 24. 
    * There is a **large jump from the 50th percentile (5) and 75th percentile (24)**. Speculating that perhaps there are 2 cathegories of houses, those in rural areas and those more urban. 
* **TAX:** Full-value property-tax rate per 10,000 dollars
    * Ranges from 187-711 with and average of 408. Distribution looks nominal.
    * **That range suggests these are mid to high income houses.** 
* **PTRATIO:** Pupil-teacher ratio by town
    * Ranges from 12.6-22 with an avergage of 18.4. Distribution looks nominal.
* **LSTAT:** % lower status of the population
    * Ranges from 7-37.9% with an average of 12%. This indicates that most areas have little lower socio-economic class.
    * **The jump from 75th percentile (16.9%) to the max (37%) is indicative of a lower socio-economic area or less likely an outlier**
* **MEDV:** Median value of owner-occupied homes in 1000 dollars
    * Ranges from 5k-50k with an average of 22. Distribution looks nominal.

### **Univariate Analysis**

**Observations:**
* **CRIM:** Per capita crime rate by town
    * Heavily right skewed with most values being 0.
* **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.
    * Most residential areas have 0 ZN, followed by a near uniform distribution from 10-100%
* **INDUS:** Proportion of non-retail business acres per town
    * Apears to be 2 peaks centered at 5% and 17%.
* **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    * Very few houses tract river
* **NOX:** Nitric Oxide concentration (parts per 10 million)
    * Right skewed
* **RM:** The average number of rooms per dwelling
    * Reletively normal distribution around 6.2
* **AGE:** Proportion of owner-occupied units built before 1940
    * Heavily left-skewed, **suggesting most hases are older**
* **DIS:** Weighted distances to five Boston employment centers
    * Heavily right-skewed
* **RAD:** Index of accessibility to radial highways
    * Reiterrates our above observation, likelke **two cathegories of houses (rural and urban)**.
* **TAX:** Full-value property-tax rate per 10,000 dollars
    * Again looks like a similar representation to RAD of **two cathegories of houses (rural and urban)**.
* **PTRATIO:** Pupil-teacher ratio by town
    * Left-skewed
* **LSTAT:** % lower status of the population
    * Right-skewed suggesting there are fewer overall lower socio-economic people.
* **MEDV:** Median value of owner-occupied homes in 1000 dollars
    * Slightly skewed. **As this is our dependent variable will need to take action to normalize it**.

Least squares regression models assume the residuals are normal, and a non-normal dependent variable will produce non-normal residual errors. Therefore, as the dependent variable is sightly skewed, we need to apply a **log transformation on the 'MEDV' column** and check the distribution of the transformed column.

Note: Using methods like quantile regression and robust regression can use non-normal dependent variables.

**Observation:**

The log-transformation (**MEDV_log**) appears to have a **nearly normal distribution without skew**, therefore we can proceed.

### **Bivariate Analysis**

**Check the correlation using heatmap**

**Observations:**

**Correlations involving dependent variable:**
* The highest possitive correlating feature for `MEDV_log` is `RM`(average number of rooms).
    * This makes sense as more rooms typically indicates a larger home
* The highest negative correlating feature for `MEDV_log` is `LSTAT`(% lower status of the population).
    * This makes sense as cities often have lower income areas. 
* It is note worthy that 8/12 of our features have negative correlations with `MEDV_log`, this means **most of them are measuring undesirable factors**.
---------------------
**Other strong correlations (>= 0.7 or <= -0.7) not involving our dependent variable:**
* Positive Correlation between `NOX` and `INDUS`, makes sense as more industrial areas would produces more Nitric Oxide
* Positive Correlation between `NOX` and `AGE`, perhaps indicating that the older areas are more industrialized?
* Neggative Correlation between `DIS` and `INDUS`, `DIS` and `NOX`, `DIS` and `AGE`. 
    * Distance to Boston employment centers seems to indicate a more modern area seperate from the older industrial areas that produce more nitric oxide.
* Positive Correlation between `TAX` and `INDUS`
* Very high Positive Correlation between `TAX` and `RAD`

#### **Visualizing the relationship between the features having significant correlations (>= 0.7 or <= -0.7)**

**Observations:**
- The distance of the houses to the Boston employment centers appears to decrease moderately as the the proportion of the old houses increase in the town. It is possible that the Boston employment centers are located in the established towns where proportion of owner-occupied units built prior to 1940 is comparatively high.

**Observations:**

- The correlation between RAD and TAX is very high. But, no trend is visible between the two variables. 
- **The strong correlation might be due to outliers.**

Check the correlation remains after removing the outliers.

**Observation:**

- So, the high correlation between TAX and RAD is due to the outliers. The tax rate for some properties might be higher due to some other reason.

**Observations:**

- The tax rate appears to increase with an increase in the proportion of non-retail business acres per town. This might be due to the reason that the variables TAX and INDUS are related with a third variable.

**Observations:**

- The price of the house seems to increase as the value of RM increases. This is expected as the price is generally higher for more rooms.

- There are a few outliers in a horizontal line as the MEDV value seems to be capped at 50.

**Observations:**

- The price of the house tends to decrease with an increase in LSTAT. This is also possible as the house price is lower in areas where lower status people live.
- There are few outliers and the data seems to be capped at 50.

**Observations:**
* Nitric Oxide does seem to increase with industrial areas
* No obviouse outliers present

**Observations:**
* Slight increase in Nitric Oxide with age of the house, againg giving credence to the theory that those are more industrial areas
* Posiblibly a group of highest NOX values being outliers.

**Observations:**
* Nitric Oxide strongly decreases with distance to employment centers. Possible that those centers are located in newer less industruse parts of Boston.

LSTAT and RM have a linear relationship with the dependent variable MEDV. Also, there are significant **relationships among few independent variables, which is not desirable for a linear regression model**. 

Let's first split the dataset.

## **Split the dataset**

Let's split the data into the dependent and independent variables and further split it into train and test set in a ratio of 70:30 for train and test sets.

**Intercept Term**

Allows the regression line to be shifted up or down on the y-axis to better fit the data. The value of the intercept term can be interpreted as the expected value of the dependent variable when all independent variables are set to zero.

check the multicollinearity in the training dataset.

### **Check for Multicollinearity**

Using the Variance Inflation Factor (VIF), to check if there is multicollinearity in the data.

Features having a VIF score > 5 will be dropped / treated till all the features have a VIF score < 5

**Observations:**

- There are two variables with a high VIF - RAD and TAX (greater than 5). 
- Let's remove TAX as it has the highest VIF values and check the multicollinearity again.

#### Drop the column 'TAX' from the training data and check if multicollinearity is resolved.

VIF is less than 5 for all the independent variables, and we can assume that multicollinearity has been removed between the variables.

## **Model Building**

### **Linear Regression Model1**

**Observations:**
* R-squared assesment is not bad at 76.9%, can be improved

#### **Examining the significance of the model variables**

It is not enough to fit a multiple regression model to the data, it is necessary to check whether all the regression coefficients are significant or not. Significance here means whether the population regression parameters are significantly different from zero. 

From the above it may be noted that the regression coefficients corresponding to ZN, AGE, and INDUS are not statistically significant at level α = 0.05. In other words, the regression coefficients corresponding to these three are not significantly different from 0 in the population. Hence, we will eliminate the three features and create a new model.

### **Model2 - Using significant variables**

Now, we will check the linear regression assumptions.

#### **Checking the below linear regression assumptions**

1. **Mean of residuals should be 0**
2. **No Heteroscedasticity**
3. **Linearity of variables**
4. **Normality of error terms**

##### **1. Check for mean residuals**

**Observations:**
* The mean residuals is very close to 0, therefore **the assumption is satisfied.**

##### **2. Check for homoscedasticity**

- Homoscedasticity - If the residuals are symmetrically distributed across the regression line, then the data is said to be homoscedastic.

- Heteroscedasticity- - If the residuals are not symmetrically distributed across the regression line, then the data is said to be heteroscedastic. In this case, the residuals can form a funnel shape or any other non-symmetrical shape.

- We'll use `Goldfeldquandt Test` to test the following hypothesis with alpha = 0.05:

    - Null hypothesis: Residuals are homoscedastic
    - Alternate hypothesis: Residuals have heteroscedastic

**Observations:**
* Since the p-value < 0.05, we reject the Null-Hypothesis hence residuals have heteroscedastic. 
* **Therefore the assumption is not satisfied and our model will overall be less accurate.**
* We can try and solve this by further transforming Y.

##### **3. Linearity of variables**

It states that the predictor variables must have a linear relation with the dependent variable.

To test the assumption, we'll plot residuals and the fitted values on a plot and ensure that residuals do not form a strong pattern. They should be randomly and uniformly scattered on the x-axis.

**Observations:**
* There is no pattern in the residual vs fitted values, therefore **the assumption is satesfied**.

##### **4. Normality of error terms**

The residuals should be normally distributed.

**Observations:**
* From the above plots, the residuals are skewed.
* **Therefore the assumption is not satisfied and our model will overall be less accurate.**

### **Check the performance of the model on the train and test data set**

**Observations:**
* The train and test scores are very close, therefore our model **is not overfitted and generalizes well**.
* That the two scores are so close means there is likely little we can do to improve the model performance.

### **Apply cross validation to improve the model and evaluate it using different evaluation metrics**

**Observations**
* As predicted the model is already at peak performance and did not improve.

### Get model Coefficients 
Put model coefficients in a pandas dataframe with column 'Feature' having all the features and column 'Coefs' with all the corresponding Coefs. (4 Marks)

**Hint:** To get values please use coef.values

## **Conclusions and Business Recommendations**

### **Conclusions**

* We can use this forecasting model to predict the housing prices in Boston.
* The model explains 100% of the variation in the data with an r-squared of 1.
* The top 5 features that have the greatest impact on predicting housing prices are:
    * CRIM: Per capita crime rate by town - Where a lower crime rate results in a higher prices.
    * CHAS: Charles River dummy variable - Where being on the Charles River results in a higher prices.
    * NOX: Nitric Oxide concentration (parts per 10 million) - Where higher nitric oxide concentration results in higher prices
        * Note that NOX was heavily correlated to INDST and AGE which where dropped for that reason. Therefore it is likely that a higher NOX is acting as a stand in for the older and more industrial areas and that is key to increasing the price.
    * RM: The average number of rooms per dwelling - Where more rooms results in a higher price
    * DIS: Weighted distances to five Boston employment centers - Where a shorter distance to employment center results in higher prices. 
        * We observed that lower DIS is likely representative of more urban areas of Boston

### **Recommendations**

Our model can very accuratly predict the housing prices in Boston and would be a usefull tool in the real estate, banking, and insurance industries. 

From our model we where able to extract that value in Boston houses is primarily measured by:
* Areas with low crime rates
* Being on the bounds of the Charles River
* Older and more industrial neighboorhoods 
* Having more rooms
* Located near more urban areas
