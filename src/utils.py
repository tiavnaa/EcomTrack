# src/utils.py
"""
Fonctions utilitaires pour visualisations et reporting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_plot_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_feature_importance(feature_importance_dict, title="Feature Importance", save=False):
    features = list(feature_importance_dict.keys())
    importance = list(feature_importance_dict.values())
    
    df_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    df_importance = df_importance.sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(df_importance['Feature'], df_importance['Importance'], color='steelblue')
    plt.title(title, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save:
        plt.savefig(f'outputs/figures/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(cm, labels=['Non Churn', 'Churn'], save=False):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    if save:
        plt.savefig('outputs/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_sales_prediction(actual, predicted, future_predictions=None, save=False):
    plt.figure(figsize=(14, 6))
    
    plt.plot(actual.index, actual.values, label='Actual Sales', color='blue', linewidth=2)
    plt.plot(predicted.index, predicted.values, label='Predicted Sales', color='red', linestyle='--', linewidth=2)
    
    if future_predictions is not None:
        future_index = range(len(actual), len(actual) + len(future_predictions))
        plt.plot(future_index, future_predictions, label='Future Forecast', color='green', linestyle=':', linewidth=2)
    
    plt.title('Sales Prediction Results', fontweight='bold')
    plt.xlabel('Time (Days)')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig('outputs/figures/sales_prediction.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_churn_risk(churn_probabilities, customer_ids, top_n=20, save=False):
    risk_df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Churn_Risk': churn_probabilities
    }).sort_values('Churn_Risk', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, top_n))
    bars = plt.barh(range(len(risk_df)), risk_df['Churn_Risk'].values, color=colors)
    plt.title(f'Top {top_n} Customers at Risk of Churn', fontweight='bold')
    plt.xlabel('Churn Probability')
    plt.yticks(range(len(risk_df)), risk_df['Customer_ID'].values)
    plt.tight_layout()
    
    if save:
        plt.savefig('outputs/figures/churn_risk.png', dpi=300, bbox_inches='tight')
    
    plt.show()