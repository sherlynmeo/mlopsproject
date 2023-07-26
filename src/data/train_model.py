import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle


customers = pd.read_csv("../../data/processed/Ecommerce_Customers_scaled.csv")
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


rf_regressor = RandomForestRegressor(random_state=42)
rf_final = rf_regressor.fit(X_train, y_train)


predictions = rf_final.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print('Model Performance')
print('Mean Absolute Error: {:0.4f}.'.format(mae))
print('Mean Squared Error: {:0.4f}.'.format(mse))
print('R^2 Score = {:0.4f}.'.format(r2))


filename = '../../models/RF_trained_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rf_final, file)
