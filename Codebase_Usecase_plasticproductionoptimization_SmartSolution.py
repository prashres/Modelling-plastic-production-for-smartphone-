#!/usr/bin/env python
# coding: utf-8

# In[1]:


## IMPORTING LIBRARIES 

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ------------------------------
# Step 0: Create fictive data
# ------------------------------
np.random.seed(144)
plastic_type = np.random.choice(['Polycarbonate', 'ABS', 'Polypropylene'], size=1000, p=[0.5,0.3,0.2])
np.random.seed(145); melt_flow_index = np.random.uniform(5, 30, 1000)
np.random.seed(146); moisture_content = np.random.uniform(0.001, 0.1, 1000)
np.random.seed(147); injection_temperature = np.random.normal(250,25,1000)
np.random.seed(148); injection_pressure = np.random.uniform(10000,20000,1000)
np.random.seed(149); mold_temperature = np.random.normal(80,20,1000)
np.random.seed(150); cooling_time = np.random.normal(25,5,1000)
np.random.seed(151); screw_speed = np.random.normal(50,10,1000)
np.random.seed(152); clamp_force = np.random.normal(100,20,1000)

# ------------------------------
# Step 1: Put into dataframe
# ------------------------------
df = pd.DataFrame({
    'PlasticType': plastic_type,
    'meltFlowIndex_Gramsper10Min': melt_flow_index,
    'MoistureContent_Percentage': moisture_content,
    'InjectionTemperature_Celcius': injection_temperature,
    'InjectionpPressure_Bar': injection_pressure,
    'MoldTemperature_Celcius': mold_temperature,
    'CoolingTime_Seconds': cooling_time,
    'ScrewSpeed_RevolutionsPerMinute': screw_speed,
    'ClampForce_Tons': clamp_force,
})

# ------------------------------
# Step 2: Create latent "process efficiency" factor
# ------------------------------
cols_to_correlate = [
    'meltFlowIndex_Gramsper10Min',
    'MoistureContent_Percentage',
    'InjectionTemperature_Celcius',
    'InjectionpPressure_Bar',
    'MoldTemperature_Celcius',
    'CoolingTime_Seconds',
    'ScrewSpeed_RevolutionsPerMinute',
    'ClampForce_Tons'
]

scaler = StandardScaler()
X = scaler.fit_transform(df[cols_to_correlate])

# Latent efficiency factor = average of standardized process parameters
latent_efficiency = X.mean(axis=1)

# ------------------------------
# Step 3: Create dependent variables
# ------------------------------
# Cycle time: moderate negative correlation
df["CycleTime_Seconds"] = 75 + 5*latent_efficiency + np.random.normal(0,2,1000)

# Scrap rate: stronger negative correlation
df["ScrapRate_Percentage"] = 2.5 - 1.5*latent_efficiency + np.random.normal(0,0.3,1000)
df["ScrapRate_Percentage"] = df["ScrapRate_Percentage"].clip(lower=0)

# ------------------------------
# Step 4: Quick check of correlations
# ------------------------------
print(df.corr()[["CycleTime_Seconds","ScrapRate_Percentage"]].loc[cols_to_correlate])


# In[ ]:


## SAVING TO DESKTOP 

import os

# File name
file_name = "smartphone_plastic_production.csv"

# Path to Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", file_name)

# Export DataFrame to CSV
df.to_csv(desktop_path, index=True)

print(f"CSV file saved to: {desktop_path}")


# In[2]:


## DESCRIPTIVE ANALYSIS OF DATA

import matplotlib.pyplot as plt 
import seaborn as sns

# 1. Basic descriptive statistics
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Stats:\n", df.describe())


# In[3]:


## MULTIVARIATE ANALYSIS 

# 3. Correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 4. Top correlations with Scrap Rate
scrap_corr = numeric_df.corr()['ScrapRate_Percentage'].sort_values(ascending=False)
print("\nTop correlations with Scrap Rate:")
print(scrap_corr.head(6))
print("\nBottom correlations with Scrap Rate:")
print(scrap_corr.tail(6))

# 4. Top correlations with Cycle Time
scrap_corr = numeric_df.corr()['CycleTime_Seconds'].sort_values(ascending=False)
print("\nTop correlations with Cycle Time:")
print(scrap_corr.head(6))
print("\nBottom correlations with Cycle Time:")
print(scrap_corr.tail(6))


# In[4]:


# DATA PREPROCESSING 

## 1. OHEHOTENCODING OF PLASTIC TYPES TO NUMERICAL 

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

oh = OneHotEncoder(sparse_output=False)  

# Fit and transform
one_hot_encoded = oh.fit_transform(df[['PlasticType']])

# Convert to DataFrame
one_hot_df = pd.DataFrame(
    one_hot_encoded, 
    columns=oh.get_feature_names_out(['PlasticType']), 
    index=df.index
)

# Concatenate with original DataFrame
df_encoded = pd.concat([df.drop('PlasticType', axis=1), one_hot_df], axis=1)


# In[5]:


df_encoded


# In[6]:


# -----------------------------
# 1. Train-Test Split for CycleTime Prediction
# -----------------------------
from sklearn.model_selection import train_test_split

# Features (exclude targets) and target variable (CycleTime_Seconds)
X = df_encoded.drop(columns=['CycleTime_Seconds', 'ScrapRate_Percentage'])
y_cycle = df['CycleTime_Seconds']

# Split into training and testing sets
X_train_CycleTime, X_test_CycleTime, y_train_CycleTime, y_test_CycleTime = train_test_split(
    X, 
    y_cycle, 
    test_size=0.2,        # 20% test data
    random_state=42       # reproducibility
)


# -----------------------------
# 2. Train-Test Split for ScrapRate Prediction
# -----------------------------

# Target variable (ScrapRate_Percentage)
y_scrap = df['ScrapRate_Percentage']

# Split into training and testing sets
X_train_ScrapRate, X_test_ScrapRate, y_train_ScrapRate, y_test_ScrapRate = train_test_split(
    X, 
    y_scrap, 
    test_size=0.2,        # 20% test data
    random_state=43       # different seed for variation
)


# In[7]:


# -----------------------------
# 3. Standard Scaling
# -----------------------------
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Scale features for CycleTime prediction
X_train_CycleTime_scaled = scaler.fit_transform(X_train_CycleTime)  # fit & transform on train
X_test_CycleTime_scaled = scaler.transform(X_test_CycleTime)        # only transform on test

# Scale features for ScrapRate prediction
X_train_ScrapRate_scaled = scaler.fit_transform(X_train_ScrapRate)  # fit & transform on train
X_test_ScrapRate_scaled = scaler.transform(X_test_ScrapRate)        # only transform on test


# In[8]:


X_train_ScrapRate_scaled


# In[9]:


# RANDOM FOREST WITH HYPERPARAMETER TUNING (GridSearchCV)

# Importing required libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------
# 1️⃣ Defining parameter grid for Random Forest
# ------------------------------------------------------------ 
param_grid = {
    'n_estimators': [100, 200, 300],          # number of trees
    'max_depth': [None, 10, 20, 30],          # maximum tree depth
    'min_samples_split': [2, 5, 10],          # min samples to split node
    'min_samples_leaf': [1, 2, 4],            # min samples at leaf node
    'max_features': ['sqrt', 'log2']          # number of features to consider for split
}

# ------------------------------------------------------------
# Grid Search for Cycle Time Prediction
# ------------------------------------------------------------
print("Running GridSearchCV for Cycle Time Prediction...")

rf_base_cycle = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search_cycle = GridSearchCV(
    estimator=rf_base_cycle,
    param_grid=param_grid,
    cv=3,                        # 3-fold cross-validation
    n_jobs=-1,                   # use all CPU cores
    verbose=1,                   # progress output
    scoring='r2'                 # optimize for R² score
)

# Fit model on training data
grid_search_cycle.fit(X_train_CycleTime_scaled, y_train_CycleTime)

# Best hyperparameters
print("Best Parameters for Cycle Time Model:")
print(grid_search_cycle.best_params_)

# Retrieve best model
rf_cycle_best = grid_search_cycle.best_estimator_

# Predictions
pred_cycle_best = rf_cycle_best.predict(X_test_CycleTime_scaled)

# Evaluation
rmse_cycle_best = np.sqrt(mean_squared_error(y_test_CycleTime, pred_cycle_best))
r2_cycle_best = r2_score(y_test_CycleTime, pred_cycle_best)

print(f"\nCycle Time_RF (Tuned) -> RMSE: {rmse_cycle_best:.3f}, R²: {r2_cycle_best:.3f}")



# In[10]:


# ------------------------------------------------------------
# 3️⃣ Grid Search for Scrap Rate Prediction
# ------------------------------------------------------------
print("\n Running GridSearchCV for Scrap Rate Prediction...")

rf_base_scrap = RandomForestRegressor(random_state=42)

grid_search_scrap = GridSearchCV(
    estimator=rf_base_scrap,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring='r2'
)

grid_search_scrap.fit(X_train_ScrapRate_scaled, y_train_ScrapRate)

# Best hyperparameters
print("\n Best Parameters for Scrap Rate Model:")
print(grid_search_scrap.best_params_)

# Retrieve best model
rf_scrap_best = grid_search_scrap.best_estimator_

# Predictions
pred_scrap_best = rf_scrap_best.predict(X_test_ScrapRate_scaled)

# Evaluation
rmse_scrap_best = np.sqrt(mean_squared_error(y_test_ScrapRate, pred_scrap_best))
r2_scrap_best = r2_score(y_test_ScrapRate, pred_scrap_best)

print(f"\nScrap Rate_RF (Tuned) -> RMSE: {rmse_scrap_best:.3f}, R²: {r2_scrap_best:.3f}")


# In[11]:


# 4️⃣ Feature Importance (from tuned models)
# ------------------------------------------------------------
# Feature importance plots help interpret key process parameters

# --- Cycle Time ---
importances_cycle = rf_cycle_best.feature_importances_
features_cycle = X_train_CycleTime.columns

plt.figure(figsize=(10, 6))
plt.barh(features_cycle, importances_cycle, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Random Forest (Tuned) - Cycle Time Feature Importance")
plt.show()

# --- Scrap Rate ---
importances_scrap = rf_scrap_best.feature_importances_
features_scrap = X_train_ScrapRate.columns

plt.figure(figsize=(10, 6))
plt.barh(features_scrap, importances_scrap, color='lightgreen')
plt.xlabel("Feature Importance")
plt.title("Random Forest (Tuned) - Scrap Rate Feature Importance")
plt.show()


# In[12]:


# ============================================================
# 🧠 OPTIMIZATION OF PROCESS PARAMETERS (USING TUNED MODELS)
# ============================================================

import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------
# 1️⃣ Define objective function
# ------------------------------------------------------------
def objective(x, model_cycle, model_scrap):
    """
    Objective:
    We aim to minimize scrap rate and maximize cycle time efficiency.
    Hence, the objective = -CycleTime + ScrapRate (to be minimized).

    Lower objective value -> Higher cycle time (better efficiency)
                             + Lower scrap rate (less waste)
    """
    x = np.array(x).reshape(1, -1)
    cycle_pred = model_cycle.predict(x)[0]
    scrap_pred = model_scrap.predict(x)[0]
    return -cycle_pred + scrap_pred  # minimize this


# ------------------------------------------------------------
# 2️⃣ Initial guess (mean of feature values)
# ------------------------------------------------------------
X_mean = X_train_CycleTime.mean().values


# ------------------------------------------------------------
# 3️⃣ Define bounds for all process parameters
# ------------------------------------------------------------
# Bounds ensure optimization stays within the real production limits.
bounds = []
for col in X_train_CycleTime.columns:
    bounds.append((X_train_CycleTime[col].min(), X_train_CycleTime[col].max()))


# ------------------------------------------------------------
# 4️⃣ Run optimization (using tuned Random Forest models)
# ------------------------------------------------------------
print("🚀 Running optimization using tuned Random Forest models...")

result_rf_best = minimize(
    objective,
    X_mean,
    args=(rf_cycle_best, rf_scrap_best),
    bounds=bounds,
    method='L-BFGS-B'
)

# Extract optimal feature values
optimal_rf_best = result_rf_best.x

print("\n✅ Optimal Process Settings (Tuned RF Models):")
for i, col in enumerate(X_train_CycleTime.columns):
    print(f"{col}: {optimal_rf_best[i]:.3f}")

# ------------------------------------------------------------
# 5️⃣ Predict optimized outcomes
# ------------------------------------------------------------
cycle_opt = rf_cycle_best.predict(optimal_rf_best.reshape(1, -1))[0]
scrap_opt = rf_scrap_best.predict(optimal_rf_best.reshape(1, -1))[0]

print(f"\n📈 Predicted Optimal Cycle Time: {cycle_opt:.3f} seconds")
print(f"📉 Predicted Optimal Scrap Rate: {scrap_opt:.3f}%")

# ------------------------------------------------------------
# 6️⃣ Interpretation Summary
# ------------------------------------------------------------
"""
🧾 Interpretation:

The optimization identifies the best combination of process parameters
(e.g., melt flow index, moisture content, injection pressure, cooling time, etc.)
to achieve maximum production efficiency — i.e., 
a high cycle time performance with minimal scrap rate.

The predicted optimal cycle time and scrap rate indicate 
the model’s suggestion for improving overall manufacturing yield 
while maintaining quality consistency in smartphone base production.
"""

