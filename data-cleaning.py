import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ecommerce_product_performance.csv")
df_cleaned = df.dropna().reset_index(drop=True)

features = ['Product_Price', 'Discount_Rate', 'Product_Rating',
            'Number_of_Reviews', 'Stock_Availability',
            'Days_to_Deliver', 'Return_Rate']

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[features]), columns=features)
