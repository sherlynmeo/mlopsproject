import pandas as pd
from sklearn.preprocessing import MinMaxScaler

customers = pd.read_csv("../../data/processed/Ecommerce_Customers_processed.csv")
def min_max_scaling(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data,columns=df.columns)
    return scaled_df 

scaled_df_min_max_score = min_max_scaling(customers)
scaled_df_min_max_score.to_csv("../../data/processed/Ecommerce_Customers_scaled.csv",sep=',',index=False)