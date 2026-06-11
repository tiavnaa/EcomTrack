import streamlit as st
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models import ChurnPredictor, SalesPredictor
from src.utils import set_plot_style, plot_feature_importance, plot_confusion_matrix

st.set_page_config(
    page_title="Predictions ML",
    page_icon="🤖",
    layout="wide"
)

st.title("Predictions Machine Learning")
st.markdown("---")

@st.cache_data
def load_and_prepare_data():
    loader = DataLoader('data/ecommerce_customer.csv')
    df = loader.load_data()
    df = loader.clean_data()
    return df

df = load_and_prepare_data()
feature_engineer = FeatureEngineer(df)

tab1, tab2 = st.tabs([" Prediction Churn Client", " Prediction Ventes Futures"])

with tab1:
    st.header("Prediction du risque de perte de client")
    
    st.markdown("""
    Cette analyse predit quels clients sont les plus susceptibles d'arreter d'acheter.
    Le modele utilise les donnees historiques pour identifier les clients a risque.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        churn_threshold = st.slider(
            "Seuil de churn (jours sans activite)",
            min_value=30,
            max_value=180,
            value=90,
            step=30
        )
    
    with col2:
        train_churn = st.button("Entrainer le modele de churn", type="primary")
    
    if train_churn:
        with st.spinner("Preparation des donnees et entrainement du modele..."):
            X, y, customer_features = feature_engineer.prepare_churn_data(churn_threshold)
            
            churn_predictor = ChurnPredictor()
            results = churn_predictor.train(X, y)
            
            st.session_state['churn_model'] = churn_predictor
            st.session_state['churn_results'] = results
            st.session_state['churn_features'] = X
            st.session_state['churn_customers'] = customer_features
    
    if 'churn_results' in st.session_state:
        results = st.session_state['churn_results']
        churn_predictor = st.session_state['churn_model']
        customer_features = st.session_state['churn_customers']
        
        st.subheader("Performance du modele")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.2%}")
        
        with col2:
            st.metric("Features utilisees", len(results['feature_importance']))
        
        st.subheader("Matrice de confusion")
        plot_confusion_matrix(results['confusion_matrix'])
        st.pyplot(plt.gcf())
        
        st.subheader("Importance des features")
        plot_feature_importance(results['feature_importance'], "Churn Feature Importance")
        st.pyplot(plt.gcf())
        
        st.subheader("Clients a risque de churn")
        
        churn_probs = churn_predictor.predict(st.session_state['churn_features'])
        
        risk_df = customer_features.copy()
        risk_df['Churn_Probability'] = churn_probs
        risk_df = risk_df.sort_values('Churn_Probability', ascending=False)
        
        st.dataframe(
            risk_df[['Customer_ID', 'Churn_Probability', 'Order_ID_count', 
                    'Revenue_sum', 'Last_Purchase_Days']].head(20),
            use_container_width=True
        )
        
        high_risk = risk_df[risk_df['Churn_Probability'] > 0.7]
        st.warning(f"⚠️ {len(high_risk)} clients sont a haut risque de churn (probabilite > 70%)")

with tab2:
    st.header("Prediction des ventes futures")
    
    st.markdown("""
    Cette analyse predit le chiffre d'affaires des prochains jours.
    Le modele utilise l'historique des ventes et les tendances saisonnieres.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider(
            "Nombre de jours a predire",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
    
    with col2:
        train_sales = st.button("Entrainer le modele de ventes", type="primary")
    
    if train_sales:
        with st.spinner("Preparation des donnees et entrainement du modele..."):
            X, y, daily_sales = feature_engineer.prepare_sales_prediction_data()
            
            sales_predictor = SalesPredictor()
            results = sales_predictor.train(X, y)
            
            daily_sales_aligned = daily_sales.iloc[-len(y):]
            
            st.session_state['sales_model'] = sales_predictor
            st.session_state['sales_results'] = results
            st.session_state['sales_data'] = daily_sales_aligned
            st.session_state['forecast_days'] = forecast_days
    
    if 'sales_results' in st.session_state:
        results = st.session_state['sales_results']
        sales_predictor = st.session_state['sales_model']
        daily_sales = st.session_state['sales_data']
        forecast_days = st.session_state['forecast_days']
        
        st.subheader("Performance du modele")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"{results['mae']:.2f} €")
        
        with col2:
            st.metric("RMSE", f"{results['rmse']:.2f} €")
        
        with col3:
            st.metric("R² Score", f"{results['r2']:.2%}")
        
        with col4:
            st.metric("Features utilisees", len(results['feature_importance']))
        
        st.subheader("Importance des features")
        plot_feature_importance(results['feature_importance'], "Sales Feature Importance")
        st.pyplot(plt.gcf())
        
        st.subheader(f"Predictions des ventes pour les {forecast_days} prochains jours")
        
        future_predictions = sales_predictor.predict_future(forecast_days, daily_sales)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        last_30_actual = daily_sales['Revenue'].tail(30)
        ax.plot(last_30_actual.index, last_30_actual.values, label='Ventes reelles', color='blue', linewidth=2)
        
        future_index = range(len(daily_sales), len(daily_sales) + forecast_days)
        ax.plot(future_index, future_predictions, label='Predictions futures', color='green', linestyle='--', linewidth=2)
        
        ax.set_title(f'Prediction des ventes - {forecast_days} jours', fontweight='bold')
        ax.set_xlabel('Jours')
        ax.set_ylabel('Chiffre d\'affaires (€)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        total_forecast = future_predictions.sum()
        avg_forecast = future_predictions.mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("CA total predit", f"{total_forecast:,.0f} €")
        
        with col2:
            st.metric("CA moyen par jour", f"{avg_forecast:.0f} €")
        
        forecast_df = pd.DataFrame({
            'Jour': range(1, forecast_days + 1),
            'CA_Predit': future_predictions
        })
        st.dataframe(forecast_df, use_container_width=True)

st.markdown("---")
st.caption("Les performances des modeles dependent de la qualite et de la quantite des donnees")