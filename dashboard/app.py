import streamlit as st
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.analysis import EcommerceAnalyzer
from src.features import FeatureEngineer
from src.models import ChurnPredictor, SalesPredictor
from src.utils import set_plot_style

st.set_page_config(
    page_title="ECOMTRACK Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_and_process_data():
    cache_key = 'ecommerce_data_processed'
    
    if cache_key not in st.session_state:
        with st.spinner("Chargement des donnees..."):
            loader = DataLoader('data/ecommerce_customer.csv')
            df = loader.load_data()
            df = loader.clean_data()
            
            analyzer = EcommerceAnalyzer(df)
            revenue_metrics = analyzer.get_revenue_analysis()
            
            st.session_state[cache_key] = {
                'df': df,
                'analyzer': analyzer,
                'revenue_metrics': revenue_metrics
            }
    
    return st.session_state[cache_key]

def main():
    st.title("ECOMTRACK Dashboard")
    st.markdown("---")
    
    data = load_and_process_data()
    df = data['df']
    analyzer = data['analyzer']
    revenue_metrics = data['revenue_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Chiffre d'affaires total", f"{revenue_metrics['total_revenue']:,.0f} €")
    
    with col2:
        st.metric("Valeur moyenne par commande", f"{revenue_metrics['avg_revenue_per_order']:.2f} €")
    
    with col3:
        st.metric("Nombre total de commandes", f"{len(df):,}")
    
    with col4:
        st.metric("Clients uniques", f"{df['Customer_ID'].nunique():,}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs([" Analyse Produits", " Analyse Clients", " Analyse Ventes"])
    
    with tab1:
        st.header("Analyse des Produits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 produits par CA")
            top_revenue = analyzer.get_top_products(10, 'revenue')
            st.dataframe(top_revenue, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 produits par quantite")
            top_quantity = analyzer.get_top_products(10, 'quantity')
            st.dataframe(top_quantity, use_container_width=True)
        
        st.subheader("Produits les moins performants")
        bottom_products = analyzer.get_bottom_products(5, 'revenue')
        st.dataframe(bottom_products, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            analyzer.plot_top_products(10, 'revenue')
            st.pyplot(plt.gcf())
        
        with col2:
            analyzer.plot_top_products(10, 'quantity')
            st.pyplot(plt.gcf())
    
    with tab2:
        st.header("Analyse des Clients")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top clients (par commandes)")
            loyal_orders = analyzer.get_loyal_customers(10, 'orders')
            st.dataframe(loyal_orders, use_container_width=True)
        
        with col2:
            st.subheader("Top clients (par CA)")
            loyal_revenue = analyzer.get_loyal_customers(10, 'revenue')
            st.dataframe(loyal_revenue, use_container_width=True)
        
        st.subheader("Segmentation RFM")
        rfm = analyzer.get_customer_segmentation()
        segment_counts = rfm['Segment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(segment_counts, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            segment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_title('Distribution des segments clients')
            st.pyplot(fig)
    
    with tab3:
        st.header("Analyse des Ventes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CA par mois")
            analyzer.plot_revenue_trend('Month')
            st.pyplot(plt.gcf())
        
        with col2:
            st.subheader("CA par jour de semaine")
            analyzer.plot_revenue_trend('Day_of_Week')
            st.pyplot(plt.gcf())
        
        st.subheader("CA par trimestre")
        analyzer.plot_revenue_trend('Quarter')
        st.pyplot(plt.gcf())
    
    st.markdown("---")
    st.caption("Dashboard développé avec Streamlit")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    set_plot_style()
    main()