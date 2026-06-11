import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Données chargées: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
            return self.df
        except FileNotFoundError:
            print(f"Fichier non trouvé: {self.file_path}")
            raise
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            raise
    
    def clean_data(self):
        if self.df is None:
            raise ValueError("Chargez d'abord les données avec load_data()")
        
        print("\n--- Nettoyage des données ---")
        
        # 1. Converti Date en datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        print("Conversion de 'Date' en datetime")
        
        # 2. Crée des colonnes temporelles
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Month_Name'] = self.df['Date'].dt.strftime('%B')
        self.df['Day'] = self.df['Date'].dt.day
        self.df['Day_of_Week'] = self.df['Date'].dt.dayofweek
        self.df['Weekday_Name'] = self.df['Date'].dt.strftime('%A')
        self.df['Quarter'] = self.df['Date'].dt.quarter
        print("Colonnes temporelles ajoutées")
        
        # 3. Vérifie les valeurs manquantes
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            print(f"Colonnes avec valeurs manquantes: {missing_cols}")
            for col in missing_cols:
                null_count = self.df[col].isnull().sum()
                print(f"   - {col}: {null_count} valeurs manquantes ({null_count/len(self.df)*100:.2f}%)")
        else:
            print("Aucune valeur manquante détectée")
        
        # 4. assure que Total_Amount est notre chiffre d'affaires
        self.df['Revenue'] = self.df['Total_Amount']
        
        # 5. Ajoute une colonne pour savoir si une réduction a été appliquée
        self.df['Has_Discount'] = (self.df['Discount_Amount'] > 0).astype(int)
        
        # 6. Ajoute une colonne pour le prix unitaire après réduction
        self.df['Unit_Price_After_Discount'] = self.df['Total_Amount'] / self.df['Quantity']
        
        # 7. Ajoute une colonne pour le taux de réduction
        self.df['Discount_Rate'] = (self.df['Discount_Amount'] / (self.df['Unit_Price'] * self.df['Quantity'])).fillna(0)
        
        print("Colonnes dérivées ajoutées (Revenue, Has_Discount, Discount_Rate)")
        
        return self.df
    
    def get_data_summary(self):
        if self.df is None:
            raise ValueError("Chargez d'abord les données avec load_data()")
        
        print("\n" + "="*60)
        print("RÉSUMÉ DES DONNÉES")
        print("="*60)
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'total_customers': self.df['Customer_ID'].nunique(),
            'total_revenue': self.df['Revenue'].sum(),
            'avg_order_value': self.df['Revenue'].mean(),
            'total_orders': len(self.df),
            'avg_quantity_per_order': self.df['Quantity'].mean(),
            'date_range': f"{self.df['Date'].min()} to {self.df['Date'].max()}"
        }
        
        print(f"Nombre total de transactions: {summary['total_rows']:,}")
        print(f"Nombre total de clients uniques: {summary['total_customers']:,}")
        print(f"Chiffre d'affaires total: {summary['total_revenue']:,.2f} €")
        print(f"Valeur moyenne par commande: {summary['avg_order_value']:.2f} €")
        print(f"Quantité moyenne par commande: {summary['avg_quantity_per_order']:.2f}")
        print(f"Période: {summary['date_range']}")
        
        return summary
    
    def save_cleaned_data(self, output_path='data/cleaned_ecommerce_data.csv'):
        if self.df is None:
            raise ValueError("Chargez d'abord les données avec load_data()")
        
        self.df.to_csv(output_path, index=False)
        print(f"Données nettoyées sauvegardées dans: {output_path}")
        return output_path
