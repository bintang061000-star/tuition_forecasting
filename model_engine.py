import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import data_prep as dp
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    datasets = [
        ('US', 0, dp.df_us, dp.df_Growth_Rent_US, dp.df_Growth_LivCost_US, dp.df_Growth_Insurance_US, dp.df_InfUS),
        ('UK', 1, dp.df_uk, dp.df_Growth_Rent_UK, dp.df_Growth_LivCost_UK, dp.df_Growth_Insurance_UK, dp.df_InfUK),
        ('Aus', 2, dp.df_au, dp.df_Growth_Rent_Aus, dp.df_Growth_LivCost_Aus, dp.df_Growth_Insurance_Aus, dp.df_InfAus),
        ('Other', 3, dp.df_other, dp.df_Growth_Rent_Other, dp.df_Growth_LivCost_Other, dp.df_Growth_Insurance_Other, dp.df_InfGlobal)
    ]
    dfs = []
    
    for nation, code, df_tui, df_rent, df_liv, df_ins, df_inf in datasets:
        merged = df_tui.merge(df_rent, on='Year', suffixes=('_tui', '_rent')) \
                       .merge(df_liv, on='Year') \
                       .merge(df_ins, on='Year', suffixes=('_liv', '_ins')) \
                       .merge(df_inf, on='Year')

        merged = merged.rename(columns={
            'Growth_tui': 'y_tuition', 
            'Growth_rent': 'y_rent',
            'Growth_liv': 'y_living',
            'Growth_ins': 'y_insur',
            'Growth': 'x_inflation'
        })
        merged['Country_Code'] = code
        dfs.append(merged)
        
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def visualize_data(df):
    # plt.figure(figsize=(10, 6))
    # sns.regplot(x='x_inflation',
    #                 y='y_living',
    #                 scatter_kws={'alpha':0.5},
    #                 data=df)
    # plt.title('Inflation vs Living Cost Growth')
    # plt.xlabel('Inflation Rate')
    # plt.ylabel('Living Cost Growth')
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.regplot(x='x_inflation',
    #                 y='y_insur',
    #                 scatter_kws={'alpha':0.5},
    #                 data=df)
    # plt.title('Inflation vs Insurance Cost Growth')
    # plt.xlabel('Inflation Rate')
    # plt.ylabel('Insurance Cost Growth')
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.regplot(x='x_inflation',
    #                 y='y_rent',
    #                 scatter_kws={'alpha':0.5},
    #                 data=df)
    # plt.title('Inflation vs Rent Cost Growth')
    # plt.xlabel('Inflation Rate')
    # plt.ylabel('Rent Cost Growth')
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.regplot(x='x_inflation',
    #                 y='y_tuition',
    #                 scatter_kws={'alpha':0.5},
    #                 data=df)
    # plt.title('Inflation vs Tuition Cost Growth')
    # plt.xlabel('Inflation Rate')
    # plt.ylabel('Tuition Cost Growth')
    # plt.grid(True)
    # plt.show()

    # Correlation Matrix
    correlation_matrix = df[['x_inflation', 'y_tuition', 'y_rent', 'y_living', 'y_insur']].corr()
    
    # Optional: Heatmap for full correlation
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Matrix')
    # plt.show()

def train_model():
    print("--- Training Model ---")
    df = load_data()
    visualize_data(df)

    print(f"Jumlah baris data (Rows): {df.shape[0]}")
    print(f"Jumlah kolom data (Columns): {df.shape[1]}")
    
    if not df.empty:
        df['x_currency'] = dp.exchange_rate_growth()
        
        X = df[['Country_Code', 'x_inflation', 'Year', 'x_currency']]
        y = df[['y_tuition', 'y_rent', 'y_living', 'y_insur']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        
        print(f"MAE: {mean_absolute_error(y_test, pred):.2f}%")
        print(f"R2 Score: {r2_score(y_test, pred):.2f}")
        
        joblib.dump(model, 'budget_predictor_model.pkl')

if __name__ == "__main__":
    train_model()