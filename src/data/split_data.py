import pandas as pd
from sklearn.model_selection import train_test_split

customers = pd.read_csv("../../data/processed/Ecommerce_Customers_scaled.csv")

X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train.to_csv("../../data/interim/X_train.csv",sep=',',index=False)