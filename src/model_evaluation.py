from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import json

def performanceMetrices(model, X, y, y_pred=np.array([])):

    performance_metrices = {}
    if not y_pred.any():
        y_pred = model.predict(X)

    errors = abs(y_pred - y)
    mape = 100 * np.mean(errors / y)
    accuracy = 100 - mape
    performance_metrices['accuracy(mape derived)'] = accuracy
    r2 = r2_score(y, y_pred)
    performance_metrices['r2'] = r2

    n, p = X.shape
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    performance_metrices['adjusted r2'] = adj_r2

    rms = mean_squared_error(y, y_pred, squared=False)
    performance_metrices['rmse'] = rms

    with open('../submission/metrics.json', 'w') as of:
        json.dump(performance_metrices, of)
    return performance_metrices


def evaluate(model, X, y):
    y_pred = model.predict(X)
    errors = abs(y_pred - y)
    mape = 100 * np.mean(errors / y)
    accuracy = 100 - mape
    r2 = r2_score(y, y_pred)

    n, p = X.shape
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    print('Model Performance')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'R-squared: {r2:.2f}')
    print(f'Adjusted R-squared: {adj_r2:.2f}')

    return accuracy