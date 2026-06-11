import streamlit as st
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import DataLoader
from src.analysis import EcommerceAnalyzer
from src.utils import set_plot_style

st.set_page_config(
    page_title="Analyse Detaillee",
    page_icon="📈",
    layout="wide"
)

st.title("Analyse Detaillee des Donnees")
st.markdown("---")

@st.cache_data
def load_data():
    loader = DataLoader('data/ecommerce_customer.csv')
    df = loader.load_data()
    df = loader.clean_data()
    return df

df = load_data()
analyzer = EcommerceAnalyzer(df)

st.sidebar.header("Filtres")

selected_categories = st.sidebar.multiselect(
    "Categories de produits",
    options=df['Product_Category'].unique(),
    default=df['Product_Category'].unique()[:5]
)

selected_payment = st.sidebar.multiselect(
    "Mode de paiement",
    options=df['Payment_Method'].unique(),
    default=df['Payment_Method'].unique()
)

min_revenue = st.sidebar.slider(
    "Revenue minimum",
    min_value=float(df['Revenue'].min()),
    max_value=float(df['Revenue'].max()),
    value=float(df['Revenue'].min())
)

filtered_df = df[
    (df['Product_Category'].isin(selected_categories)) &
    (df['Payment_Method'].isin(selected_payment)) &
    (df['Revenue'] >= min_revenue)
]

st.subheader("Aperçu des donnees filtrees")
st.dataframe(filtered_df.head(100), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution des categories de produits")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    filtered_df['Product_Category'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_xlabel('Product Category')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("Distribution des modes de paiement")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    filtered_df['Payment_Method'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel('')
    st.pyplot(fig2)

st.subheader("Distribution des ages des clients")
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.histplot(filtered_df['Age'], bins=30, kde=True, ax=ax3, color='green')
ax3.set_xlabel('Age')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

st.subheader("Distribution des evaluations clients")
fig4, ax4 = plt.subplots(figsize=(10, 6))
filtered_df['Customer_Rating'].value_counts().sort_index().plot(kind='bar', ax=ax4, color='orange')
ax4.set_xlabel('Rating')
ax4.set_ylabel('Count')
st.pyplot(fig4)

st.subheader("Correlation entre metriques")
numeric_cols = ['Age', 'Unit_Price', 'Quantity', 'Discount_Amount', 'Revenue', 
                'Session_Duration_Minutes', 'Pages_Viewed', 'Delivery_Time_Days', 'Customer_Rating']

corr_matrix = filtered_df[numeric_cols].corr()

fig5, ax5 = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5, fmt='.2f')
ax5.set_title('Matrice de correlation')
st.pyplot(fig5)

st.subheader("Revenue par genre")
gender_revenue = filtered_df.groupby('Gender')['Revenue'].agg(['sum', 'mean', 'count']).round(2)
st.dataframe(gender_revenue, use_container_width=True)

st.subheader("Revenue par type d'appareil")
device_revenue = filtered_df.groupby('Device_Type')['Revenue'].agg(['sum', 'mean', 'count']).round(2)
st.dataframe(device_revenue, use_container_width=True)

st.subheader("Impact du temps de livraison sur les evaluations")
fig6, ax6 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='Customer_Rating', y='Delivery_Time_Days', ax=ax6, palette='Set2')
ax6.set_xlabel('Customer Rating')
ax6.set_ylabel('Delivery Time (days)')
st.pyplot(fig6)