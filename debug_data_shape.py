
import pandas as pd
import data_prep as dp

def analyze_shapes():
    datasets = [
        ('US', 0, dp.df_us, dp.df_Growth_Rent_US, dp.df_Growth_LivCost_US, dp.df_Growth_Insurance_US, dp.df_InfUS),
        ('UK', 1, dp.df_uk, dp.df_Growth_Rent_UK, dp.df_Growth_LivCost_UK, dp.df_Growth_Insurance_UK, dp.df_InfUK),
        ('Aus', 2, dp.df_au, dp.df_Growth_Rent_Aus, dp.df_Growth_LivCost_Aus, dp.df_Growth_Insurance_Aus, dp.df_InfAus),
        ('Other', 3, dp.df_other, dp.df_Growth_Rent_Other, dp.df_Growth_LivCost_Other, dp.df_Growth_Insurance_Other, dp.df_InfGlobal)
    ]
    
    total_rows = 0
    
    for nation, code, df_tui, df_rent, df_liv, df_ins, df_inf in datasets:
        print(f"\n--- {nation} ---")
        print(f"Tuition: {df_tui.shape}, Columns: {df_tui.columns.tolist()}")
        print(f"Rent: {df_rent.shape}, Columns: {df_rent.columns.tolist()}")
        print(f"Living: {df_liv.shape}, Columns: {df_liv.columns.tolist()}")
        print(f"Insurance: {df_ins.shape}, Columns: {df_ins.columns.tolist()}")
        print(f"Inflation: {df_inf.shape}, Columns: {df_inf.columns.tolist()}")
        
        merged = df_tui.merge(df_rent, on='Year', suffixes=('_tui', '_rent')) \
                       .merge(df_liv, on='Year') \
                       .merge(df_ins, on='Year', suffixes=('_liv', '_ins')) \
                       .merge(df_inf, on='Year')
        
        print(f"Merged Shape: {merged.shape}")
        print(f"Merged Columns: {merged.columns.tolist()}")
        total_rows += merged.shape[0]

    print(f"\nTotal Rows Expected: {total_rows}")

if __name__ == "__main__":
    analyze_shapes()
