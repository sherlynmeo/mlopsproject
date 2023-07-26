import pandas as pd
import numpy as np

customers = pd.read_csv("../../data/raw/Ecommerce-Customers.csv")
customers = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership', 'Yearly Amount Spent']]
customers.to_csv("../../data/processed/Ecommerce_Customers_processed.csv",sep=',',index=False)
