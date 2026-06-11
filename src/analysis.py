import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class EcommerceAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        
    def get_top_products(self, n=10, metric='revenue'):
        if metric == 'revenue':
            top = self.df.groupby('Product_Category')['Revenue'].sum().sort_values(ascending=False).head(n)
            title = f"Top {n} des catégories par chiffre d'affaires"
        elif metric == 'quantity':
            top = self.df.groupby('Product_Category')['Quantity'].sum().sort_values(ascending=False).head(n)
            title = f"Top {n} des catégories par quantité vendue"
        elif metric == 'orders':
            top = self.df.groupby('Product_Category').size().sort_values(ascending=False).head(n)
            title = f"Top {n} des catégories par nombre de commandes"
        else:
            raise ValueError("metric doit être 'revenue', 'quantity' ou 'orders'")
        
        result_df = pd.DataFrame({title: top})
        self.results[f'top_products_{metric}'] = result_df
        return result_df
    
    def get_bottom_products(self, n=5, metric='revenue'):
        if metric == 'revenue':
            bottom = self.df.groupby('Product_Category')['Revenue'].sum().sort_values().head(n)
            title = f"Bottom {n} des catégories par chiffre d'affaires"
        elif metric == 'quantity':
            bottom = self.df.groupby('Product_Category')['Quantity'].sum().sort_values().head(n)
            title = f"Bottom {n} des catégories par quantité vendue"
        else:
            raise ValueError("metric doit être 'revenue' ou 'quantity'")
        
        result_df = pd.DataFrame({title: bottom})
        self.results[f'bottom_products_{metric}'] = result_df
        return result_df
    
    def get_loyal_customers(self, n=10, metric='orders'):
        if metric == 'orders':
            loyal = self.df.groupby('Customer_ID').size().sort_values(ascending=False).head(n)
            title = f"Top {n} clients par nombre de commandes"
        elif metric == 'revenue':
            loyal = self.df.groupby('Customer_ID')['Revenue'].sum().sort_values(ascending=False).head(n)
            title = f"Top {n} clients par chiffre d'affaires"
        else:
            raise ValueError("metric doit être 'orders' ou 'revenue'")
        
        result_df = pd.DataFrame({title: loyal})
        self.results[f'loyal_customers_{metric}'] = result_df
        return result_df
    
    def get_revenue_analysis(self):
        revenue_metrics = {
            'total_revenue': self.df['Revenue'].sum(),
            'avg_revenue_per_order': self.df['Revenue'].mean(),
            'median_revenue_per_order': self.df['Revenue'].median(),
            'min_revenue': self.df['Revenue'].min(),
            'max_revenue': self.df['Revenue'].max(),
            'revenue_std': self.df['Revenue'].std(),
            'total_discount_given': self.df['Discount_Amount'].sum(),
            'avg_discount_rate': self.df['Discount_Rate'].mean() * 100,
            'orders_with_discount': self.df['Has_Discount'].sum(),
            'pct_orders_with_discount': (self.df['Has_Discount'].sum() / len(self.df)) * 100
        }
        
        self.results['revenue_analysis'] = revenue_metrics
        return revenue_metrics
    
    def get_revenue_by_time(self, period='Month'):
        if period == 'Month':
            revenue_by_time = self.df.groupby('Month_Name')['Revenue'].sum()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            revenue_by_time = revenue_by_time.reindex([m for m in month_order if m in revenue_by_time.index])
        elif period == 'Quarter':
            revenue_by_time = self.df.groupby('Quarter')['Revenue'].sum()
        elif period == 'Day_of_Week':
            revenue_by_time = self.df.groupby('Weekday_Name')['Revenue'].sum()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            revenue_by_time = revenue_by_time.reindex([d for d in day_order if d in revenue_by_time.index])
        else:
            revenue_by_time = self.df.groupby(period)['Revenue'].sum()
        
        self.results[f'revenue_by_{period}'] = revenue_by_time
        return revenue_by_time
    
    def get_customer_segmentation(self):
        reference_date = self.df['Date'].max() + timedelta(days=1)
        
        rfm = self.df.groupby('Customer_ID').agg({
            'Date': lambda x: (reference_date - x.max()).days,  # Recency
            'Order_ID': 'count',  # Frequency
            'Revenue': 'sum'  # Monetary
        }).rename(columns={
            'Date': 'Recency',
            'Order_ID': 'Frequency',
            'Revenue': 'Monetary'
        })
        
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=['4', '3', '2', '1'])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4'])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=['1', '2', '3', '4'])
        
        # Score RFM combiné
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        rfm['RFM_Total'] = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int).sum(axis=1)
        
        # Segmentation
        def segment_customer(row):
            if row['RFM_Total'] >= 10:
                return 'Champions'
            elif row['RFM_Total'] >= 8:
                return 'Loyal Customers'
            elif row['RFM_Total'] >= 6:
                return 'Potential Loyalists'
            elif row['RFM_Total'] >= 4:
                return 'At Risk'
            else:
                return 'Lost'
        
        rfm['Segment'] = rfm.apply(segment_customer, axis=1)
        
        self.results['rfm_segmentation'] = rfm
        return rfm
    
    def plot_top_products(self, n=10, metric='revenue', save=False):
        top = self.get_top_products(n, metric)
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        bars = plt.bar(range(len(top)), top.iloc[:, 0].values, color=colors)
        plt.title(f'Top {n} Products by {metric.capitalize()}', fontsize=14, fontweight='bold')
        plt.xlabel('Product Category', fontsize=12)
        plt.ylabel(f'Total {metric.capitalize()}', fontsize=12)
        plt.xticks(range(len(top)), top.index, rotation=45, ha='right')
        
        for bar, value in zip(bars, top.iloc[:, 0].values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{value:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'outputs/figures/top_products_{metric}.png', dpi=300, bbox_inches='tight')
            print(f" Graphique sauvegardé: outputs/figures/top_products_{metric}.png")
        
        plt.show()
        
    def plot_revenue_trend(self, period='Month', save=False):
        revenue_trend = self.get_revenue_by_time(period)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(revenue_trend)), revenue_trend.values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        plt.fill_between(range(len(revenue_trend)), revenue_trend.values, alpha=0.3, color='#2E86AB')
        plt.title(f'Revenue Trend by {period}', fontsize=14, fontweight='bold')
        plt.xlabel(period, fontsize=12)
        plt.ylabel('Total Revenue (€)', fontsize=12)
        plt.xticks(range(len(revenue_trend)), revenue_trend.index, rotation=45)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        for i, (idx, value) in enumerate(revenue_trend.items()):
            plt.annotate(f'{value:,.0f}€', (i, value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'outputs/figures/revenue_trend_{period}.png', dpi=300, bbox_inches='tight')
            print(f" Graphique sauvegardé: outputs/figures/revenue_trend_{period}.png")
        
        plt.show()
