from sklearn.metrics import mean_squared_error, mean_absolute_error

# Evaluate the model using MSE and MAE
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae
