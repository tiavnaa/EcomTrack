import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def create_customer_features(self):
        customer_features = self.df.groupby('Customer_ID').agg({
            'Order_ID': 'count',
            'Revenue': ['sum', 'mean'],
            'Quantity': 'sum',
            'Date': ['min', 'max'],
            'Discount_Amount': 'sum',
            'Session_Duration_Minutes': 'mean',
            'Pages_Viewed': 'mean',
            'Customer_Rating': 'mean',
            'Delivery_Time_Days': 'mean'
        })
        
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
        customer_features = customer_features.reset_index()
        
        customer_features['Tenure_Days'] = (customer_features['Date_max'] - customer_features['Date_min']).dt.days
        
        customer_features['Avg_Order_Value'] = customer_features['Revenue_sum'] / customer_features['Order_ID_count']
        customer_features['Avg_Items_Per_Order'] = customer_features['Quantity_sum'] / customer_features['Order_ID_count']
        
        customer_features['Is_Returning'] = (customer_features['Order_ID_count'] > 1).astype(int)
        
        return customer_features
    
    def create_time_features(self):
        time_df = self.df.copy()
        time_df['Year'] = time_df['Date'].dt.year
        time_df['Month'] = time_df['Date'].dt.month
        time_df['Day'] = time_df['Date'].dt.day
        time_df['DayOfWeek'] = time_df['Date'].dt.dayofweek
        time_df['WeekOfYear'] = time_df['Date'].dt.isocalendar().week
        time_df['IsWeekend'] = (time_df['DayOfWeek'] >= 5).astype(int)
        
        daily_sales = time_df.groupby('Date').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        
        return daily_sales
    
    def prepare_churn_data(self, churn_threshold_days=90):
        customer_features = self.create_customer_features()
        
        last_date = self.df['Date'].max()
        
        customer_features['Last_Purchase_Days'] = (last_date - customer_features['Date_max']).dt.days
        
        customer_features['Churn'] = (customer_features['Last_Purchase_Days'] > churn_threshold_days).astype(int)
        
        feature_cols = ['Order_ID_count', 'Revenue_sum', 'Revenue_mean', 'Quantity_sum',
                       'Session_Duration_Minutes_mean', 'Pages_Viewed_mean', 
                       'Customer_Rating_mean', 'Delivery_Time_Days_mean',
                       'Tenure_Days', 'Avg_Order_Value', 'Avg_Items_Per_Order']
        
        X = customer_features[feature_cols]
        y = customer_features['Churn']
        
        return X, y, customer_features
    
    def prepare_sales_prediction_data(self, days_ahead=30):
        daily_sales = self.create_time_features()
        
        daily_sales['Revenue_Lag1'] = daily_sales['Revenue'].shift(1)
        daily_sales['Revenue_Lag7'] = daily_sales['Revenue'].shift(7)
        daily_sales['Revenue_Lag30'] = daily_sales['Revenue'].shift(30)
        
        daily_sales['Revenue_Rolling_Mean_7'] = daily_sales['Revenue'].rolling(window=7).mean()
        daily_sales['Revenue_Rolling_Mean_30'] = daily_sales['Revenue'].rolling(window=30).mean()
        
        daily_sales['DayOfWeek_sin'] = np.sin(2 * np.pi * daily_sales['DayOfWeek'] / 7)
        daily_sales['DayOfWeek_cos'] = np.cos(2 * np.pi * daily_sales['DayOfWeek'] / 7)
        
        daily_sales['WeekOfYear_sin'] = np.sin(2 * np.pi * daily_sales['WeekOfYear'] / 52)
        daily_sales['WeekOfYear_cos'] = np.cos(2 * np.pi * daily_sales['WeekOfYear'] / 52)
        
        daily_sales = daily_sales.dropna()
        
        feature_cols = ['Quantity', 'Order_ID_count', 'DayOfWeek', 'IsWeekend',
                       'Revenue_Lag1', 'Revenue_Lag7', 'Revenue_Lag30',
                       'Revenue_Rolling_Mean_7', 'Revenue_Rolling_Mean_30',
                       'DayOfWeek_sin', 'DayOfWeek_cos', 'WeekOfYear_sin', 'WeekOfYear_cos']
        
        X = daily_sales[feature_cols]
        y = daily_sales['Revenue']
        
        return X, y, daily_sales
    
    def get_customer_product_preferences(self):
        preferences = self.df.groupby(['Customer_ID', 'Product_Category']).agg({
            'Order_ID': 'count',
            'Quantity': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        preferences = preferences.sort_values(['Customer_ID', 'Revenue'], ascending=[True, False])
        
        top_preferences = preferences.groupby('Customer_ID').first().reset_index()
        
        return top_preferences[['Customer_ID', 'Product_Category', 'Revenue']]