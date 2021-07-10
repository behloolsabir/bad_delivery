# Dropoff Distance Predictor

## Requirements
---
Environment Details in [environment.yml](environment.yml). <br/>
Follow this to create the environment. 
```bash
conda env create --name delivery -f environment.yaml --force
```
## What is this?
---
This is regression model to predict the continuous target variable dropoff distance. 

## How to use this?
---
Refer [main.ipynb](src/main.ipynb) for main script. <br/>
Dependent variables are available in [config.toml](config.toml) <br/>
Exploratory analysis can be found here: [eda_report.html](data/processed_data/eda_report.html) <br/>
Discretion advised: A few of the steps is going to be time consuming.
## Output 
---
Monitoring MAPE, RMSE, R<sup>2</sup> and Adjusted R<sup>2</sup> as our performance metrics. Performance on the hold out data can be found here: [metrics.json](submission/metrics.json)

## Next Steps
---
* Use advanced models such as gradient boosted regression trees. 
* Use Bayesian hyperparameter tuning
* Automated feature engineering using Deep Feature Synthesis. (It takes at least several hours to run)
* Treat missing pincode values as boolean of present or missing.
* Context dependent outliers analysis. We can get directions from IQR method or z-score method. 
* Model error analysis. Slicing and dicing at different feature cuts. Residual distribution analysis. 
