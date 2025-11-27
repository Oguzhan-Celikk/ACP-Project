import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# --- Phase 1: Data Acquisition ---
def load_data():
    """Loads the three datasets."""
    print("Loading datasets...")
    try:
        df_high = pd.read_csv('water-level_turbidity-high.csv')
        df_medium = pd.read_csv('water-level_turbidity-medium.csv')
        df_low = pd.read_csv('water-level_turbidity-low.csv')
        return df_high, df_medium, df_low
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None

# --- Phase 2: Feature Engineering ---
def process_dataset(df, turbidity_label):
    """Cleans, tags, and normalizes a single dataset."""
    # 1. Data Cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    # 2. Tag Turbidity Levels
    df['turbidity_category'] = turbidity_label
    
    # 3. Feature Selection (Standardizing columns)
    # Expected columns based on inspection: 
    # 'ir_value', 'us_value', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'water_level'
    # We will keep 'turbidity_category' for now as well.
    
    features = ['ir_value', 'us_value', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    target = 'water_level'
    
    # Ensure all required columns exist
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
            
    # 4. Normalization (Scaling numerical features)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, scaler

def prepare_master_dataset(df_high, df_medium, df_low):
    """Merges processed datasets into a master dataframe."""
    print("Processing datasets...")
    # Process each dataset
    
    proc_high, _ = process_dataset(df_high, 'High')
    proc_medium, _ = process_dataset(df_medium, 'Medium')
    proc_low, _ = process_dataset(df_low, 'Low')
    
    # Merge
    master_df = pd.concat([proc_high, proc_medium, proc_low], ignore_index=True)
    
    return master_df

# --- Phase 3: Model Development ---
def train_models(X_train, X_test, y_train, y_test, X_class_train, X_class_test, y_class_train, y_class_test):
    """Trains and evaluates models."""
    
    # A. Auxiliary Water Turbidity Classifier
    print("\nTraining Turbidity Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_class_train, y_class_train)
    
    y_pred_class = clf.predict(X_class_test)
    acc = accuracy_score(y_class_test, y_pred_class)
    print(f"Turbidity Classifier Accuracy: {acc:.4f}")
    
    joblib.dump(clf, 'turbidity_classifier.pkl')
    print("Saved turbidity_classifier.pkl")

    # B. Water Level Regressors
    print("\nTraining Water Level Regressors...")
    regressors = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=20, random_state=42),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
        "MLP Regressor": MLPRegressor(random_state=42, max_iter=2000)
    }
    
    results = {}
    best_model_name = ""
    best_score = -float('inf') # R2 can be negative
    best_model = None
    
    for name, model in regressors.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}
        print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_model = model
            
    print(f"\nBest Regressor: {best_model_name} with R2: {best_score:.4f}")
    joblib.dump(best_model, 'best_regressor.pkl')
    print("Saved best_regressor.pkl")
    
    return results

def main():
    # 1. Load Data
    df_high, df_medium, df_low = load_data()
    if df_high is None: return

    # 2. Feature Engineering    
    # Step 1 & 2: Clean and Tag
    df_high = df_high.dropna().drop_duplicates()
    df_high['turbidity_category'] = 'High'
    
    df_medium = df_medium.dropna().drop_duplicates()
    df_medium['turbidity_category'] = 'Medium'
    
    df_low = df_low.dropna().drop_duplicates()
    df_low['turbidity_category'] = 'Low'

    # Balancing
    min_len = min(len(df_high), len(df_medium), len(df_low))
    print(f"Balancing data to {min_len} samples per class...")
    df_high = df_high.sample(n=min_len, random_state=42)
    df_medium = df_medium.sample(n=min_len, random_state=42)
    df_low = df_low.sample(n=min_len, random_state=42)
    
    # Step 3: Merge
    master_df = pd.concat([df_high, df_medium, df_low], ignore_index=True)
    
    # Step 4: Feature Selection & Scaling
    # Step 4: Feature Selection & Scaling
    features_all = ['ir_value', 'us_value', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    target = 'water_level'
    
    X = master_df[features_all]
    y = master_df[target]
    y_class = master_df['turbidity_category']
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create separate datasets
    # X_class uses ALL features (including IR) for high accuracy
    X_class = X_scaled 
    # X_reg excludes IR (index 0) for realistic R2
    X_reg = X_scaled[:, 1:] 
    
    # Save scaler for deployment
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved scaler.pkl")
    
    # Train/Test Split
    # Split both datasets simultaneously to keep indices aligned
    X_reg_train, X_reg_test, X_class_train, X_class_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X_reg, X_class, y, y_class, test_size=0.2, random_state=42
    )
    
    # 3. Model Development
    # Pass X_reg for regressors, X_class for classifier
    train_models(X_reg_train, X_reg_test, y_train, y_test, X_class_train, X_class_test, y_class_train, y_class_test)

if __name__ == "__main__":
    main()

