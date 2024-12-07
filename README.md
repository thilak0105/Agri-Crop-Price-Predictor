# Agri Crop Price Predictor

This project aims to predict future crop prices using historical data such as existing prices, rainfall, temperature, and soil fertility. The model helps stakeholders, especially governments, in making informed decisions about buffer stock release timings.

 ![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

## Project Objective
The primary goal of this project is to build a predictive model that:
- Analyzes historical agricultural data.
- Predicts the modal price of crops based on various factors.
- Supports decision-making processes in agriculture and market regulation.

## Features of the Notebook
- Data preprocessing steps for feature engineering (e.g., handling missing values, encoding categorical features).
- Exploratory Data Analysis (EDA) to understand key patterns and trends.
- Implementation of a **Random Forest Regressor** for price prediction.
- Hyperparameter tuning to improve the accuracy of the predictions.

## Tools and Techniques Used
- **Programming Language**: Python
- **Libraries**:
  - pandas
  - numpy
  - sklearn
  - matplotlib
  - seaborn
- **Model**: Random Forest Regressor

## Installation
### Steps to Use the Notebook

1. Import the required data sets:
   ```bash
   pd.read_csv()
   ```
2. Correct the Path:
   
3. Run the notebook:
   ```bash
   CPP.ipynb
   ```

## Key Features in the Notebook
1. **Feature Selection**
   - Selected features: `commodity_name`, `state`, `district`, `market`, `month_column`, `season_names`, `day`.
2. **Model Training**
   - Implemented Random Forest Regressor with hyperparameter tuning.
   - Parameters like `n_estimators`, `max_depth`, and `min_samples_split` are tuned to achieve better results.
3. **Evaluation**
   - Metrics such as R² score and feature importance are used to evaluate model performance.

## Usage/Examples
### Random Forest Implementation
```python
from sklearn.ensemble import RandomForestRegressor

# Train the model
regr = RandomForestRegressor(n_estimators=500, max_depth=50, random_state=42)
regr.fit(Xtrain, Ytrain)

# Predict and evaluate
y_pred = regr.predict(Xtest)
from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(Ytest, y_pred)}")
```


## Outcome
The model provides accurate crop price predictions, assisting the government and other stakeholders in planning buffer stock releases effectively.



## Roadmap
- Implement additional machine learning models.
- Enhance the user interface for easier interaction with the notebook.
- Expand the dataset with more features for improved predictions.
- 
## Contributing
We welcome contributions! Please fork the repository and submit a pull request. You can also open issues for discussion.

## Contributors
- Thilak L -  [GitHub Profile](https://github.com/thilak0105)
- Teammate 1 - [Subramanian G](https://github.com/Demoncyborg07)
- Teammate 2 - [Raghul A R](https://github.com/a-steel-heart)
