import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("IndianWeatherRepository.csv")

# View the first few rows
print(df)

# Basic dataset information
print(df.info())

# Creating separate dataset for data containing numeric data for analysis
df = df.select_dtypes(include='number')
cols = df.columns.values
cols

# Removing redundant columns [such as temperature in celsius & fahrenheit]
df.drop(
    ['temperature_fahrenheit', 'wind_mph', 'pressure_in', 'precip_mm',
     'feels_like_fahrenheit', 'visibility_km', 'gust_mph'],
    axis=1, inplace=True
)
df.columns

# Handle missing values if any
df.isnull().sum()

df.hist(bins=16, figsize=(20, 15))

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

temp_weather = df

# ---------------------------------------------------------
# Scatter Plots for Air Quality Metrics vs Temperature
# ---------------------------------------------------------
air_quality_metrics = [
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone',
    'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide'
]

plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
plt.plot("temperature_celsius", "air_quality_Carbon_Monoxide", data=temp_weather,
         color="lightblue", marker='o', ms="5", ls='', label="CO")
plt.xlabel("Temperature")
plt.ylabel("CO")

plt.subplot(2, 2, 2)
plt.plot("temperature_celsius", "air_quality_Ozone", data=temp_weather,
         color="r", marker='*', ms="5", ls='', label="Ozone")
plt.xlabel("Temperature")
plt.ylabel("Ozone")

plt.subplot(2, 2, 3)
plt.plot("temperature_celsius", "air_quality_Nitrogen_dioxide", data=temp_weather,
         color="green", marker='s', ms="5", ls='', label="NO2")
plt.xlabel("Temperature")
plt.ylabel("NO2")

plt.subplot(2, 2, 4)
plt.plot("temperature_celsius", "air_quality_Sulphur_dioxide", data=temp_weather,
         color="m", marker='^', ms="5", ls='', label="SO2")
plt.xlabel("Temperature")
plt.ylabel("SO2")

plt.show()

# ---------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------
sns.set(font_scale=0.9)

corr_matrix = df.corr(method="kendall")

plt.figure(figsize=(14, 14))
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='BrBG',
                       annot_kws={"fontsize": 4}, linewidths=0.1)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 2}, pad=12)

sorted_corr_mat = corr_matrix.abs().unstack().sort_values()
sorted_corr_mat = sorted_corr_mat.to_frame(name="Correlation")

# Removing highly/least correlated data (correlation > 0.90 && < 0.05)
sorted_corr_mat = sorted_corr_mat.drop(sorted_corr_mat[sorted_corr_mat['Correlation'] > 0.95].index)
sorted_corr_mat = sorted_corr_mat.drop(sorted_corr_mat[sorted_corr_mat['Correlation'] < 0.05].index)

print("Fields with max correlation are:\n")
sorted_corr_mat[sorted_corr_mat['Correlation'] > 0.80]

plt.figure(figsize=(10, 6))
plt.scatter(df['pressure_mb'], df['cloud'], alpha=0.5)
plt.title('Pressure and Cloud Cover Relationship')
plt.xlabel('Pressure (mb)')
plt.ylabel('Cloud Cover (%)')
plt.grid()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# AQI categories
# ---------------------------------------------------------
aqi_categories = {
    1: "Good",
    2: "Moderate",
    3: "Unhealthy for Sensitive Groups",
    4: "Unhealthy",
    5: "Very Unhealthy",
    6: "Hazardous"
}

# KDE Plot for PM2.5 based on AQI Index
plt.figure(figsize=(10, 6))
for level, label in aqi_categories.items():
    if level in df['air_quality_us-epa-index'].unique():
        sns.kdeplot(df.loc[df['air_quality_us-epa-index'] == level, 'air_quality_PM2.5'], label=label)

plt.title('PM2.5 Distribution Based on Air Quality Index')
plt.xlabel('PM2.5 Concentration')
plt.ylabel('Density')
plt.grid(True, alpha=0.5)
plt.legend(title="AQI Category")
plt.show()

# Map numeric AQI index to categorical labels
df['AQI_Bucket'] = df['air_quality_us-epa-index'].map(aqi_categories)

# Create a categorical count plot for AQI Bucket
sns.catplot(x='AQI_Bucket', hue='AQI_Bucket', data=df, kind="count", height=6, aspect=2, legend=False)

plt.legend(loc='upper right', title='AQI Bucket')
plt.xticks(rotation=45)
plt.tight_layout()
plt.title('Count of AQI Categories')
plt.xlabel('Air Quality Category')
plt.ylabel('Count')
plt.show()

# ---------------------------------------------------------
# Additional scatter plots
# ---------------------------------------------------------
column_pairs = [('temperature_celsius', 'humidity'), ('wind_kph', 'pressure_mb')]

for x_column, y_column in column_pairs:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter Plot: {x_column} vs {y_column}')
    plt.tight_layout()
    plt.show()

df.columns

# Categorical columns
categorical_features = df.select_dtypes(include=['object']).columns
print("Categorical features:", categorical_features)

# ---------------------------------------------------------
# Feature / target selection
# ---------------------------------------------------------
features = ['latitude', 'longitude', 'wind_kph', 'pressure_mb',
            'air_quality_PM2.5', 'air_quality_PM10']
target = 'temperature_celsius'
print(df.columns)

temperature_data = df['temperature_celsius']
temperature_factors = df[['latitude', 'longitude', 'wind_kph', 'wind_degree',
                           'pressure_mb', 'precip_in', 'humidity', 'cloud']]

# Split the Data for Training and Testing
x_train, x_test, y_train, y_test = train_test_split(
    temperature_factors, temperature_data, test_size=0.3, random_state=0
)

# Convert continuous target values into discrete categories
y_train_class = np.digitize(y_train, bins=np.linspace(y_train.min(), y_train.max(), num=10))
y_test_class = np.digitize(y_test, bins=np.linspace(y_train.min(), y_train.max(), num=10))

# Encode the categories
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_class)
y_test_encoded = label_encoder.transform(y_test_class)

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train, y_train_encoded)

y_pred_xgb = xgb_model.predict(x_test)

model_accuracy = accuracy_score(y_test_encoded, y_pred_xgb) * 100
print("XGBoost Classifier Model Evaluation:")
print(f"Accuracy: {model_accuracy:.2f}%")
print("Classification Report (Summary):")
print(classification_report(y_test_encoded, y_pred_xgb, digits=2, zero_division=0))

# ---------------------------------------------------------
# KNN model
# ---------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train_encoded)

y_pred_knn = knn_model.predict(x_test)

model_accuracy = accuracy_score(y_test_encoded, y_pred_knn) * 100
print("KNN Classifier Model Evaluation:")
print(f"Accuracy: {model_accuracy:.2f}%")
print("Classification Report (Summary):")
print(classification_report(y_test_encoded, y_pred_knn, digits=2, zero_division=0))

# ---------------------------------------------------------
# Logistic Regression model
# ---------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg = LogisticRegression(max_iter=500, solver='lbfgs')
log_reg.fit(x_train, y_train_encoded)

y_pred_lr = log_reg.predict(x_test)

model_accuracy = accuracy_score(y_test_encoded, y_pred_lr) * 100
print("Logistic Regression Model Evaluation:")
print(f"Accuracy: {model_accuracy:.2f}%")
print("Classification Report (Summary):")
print(classification_report(y_test_encoded, y_pred_lr, digits=2, zero_division=0))

# ---------------------------------------------------------
# Random Forest model
# ---------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train_encoded)

y_pred_rf = rf_model.predict(x_test)

model_accuracy = accuracy_score(y_test_encoded, y_pred_rf) * 100
print("Random Forest Classifier Model Evaluation:")
print(f"Accuracy: {model_accuracy:.2f}%")
print("Classification Report (Summary):")
print(classification_report(y_test_encoded, y_pred_rf, digits=2, zero_division=0))

# ---------------------------------------------------------
# Model accuracy comparison chart
# ---------------------------------------------------------
import matplotlib.pyplot as plt

models = ["Random Forest", "KNN", "XGBoost", "Logistic Regression"]

# Replace these with your actual model accuracy scores
accuracies = [89.55, 77.14, 88.84, 63.61]

colors = ['#85C1E9', '#D7DBDD', '#E59866', '#E74C3C']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.2f}%",
              ha='center', fontsize=12, fontweight='bold')

plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Models", fontsize=12)
plt.ylabel("Accuracy Score (%)", fontsize=12)
plt.ylim(60, 95)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ---------------------------------------------------------
# Prediction comparison table
# ---------------------------------------------------------
print(f"Length of y_test: {len(y_test)}")
print(f"Length of y_pred_xgb: {len(y_pred_xgb)}")
print(f"Length of y_pred_rf: {len(y_pred_rf)}")
print(f"Length of y_pred_lr: {len(y_pred_lr)}")
print(f"Length of y_pred_knn: {len(y_pred_knn)}")

min_length = min(len(y_test), len(y_pred_xgb), len(y_pred_rf), len(y_pred_lr), len(y_pred_knn))

# Trim all arrays to the same length
y_test_trimmed = y_test[:min_length]
y_pred_xgb_trimmed = y_pred_xgb[:min_length]
y_pred_rf_trimmed = y_pred_rf[:min_length]
y_pred_lr_trimmed = y_pred_lr[:min_length]
y_pred_knn_trimmed = y_pred_knn[:min_length]

pred_df = pd.DataFrame({
    'Actual': y_test_trimmed,
    'Predicted (XGBoost)': y_pred_xgb_trimmed,
    'Predicted (Random Forest)': y_pred_rf_trimmed,
    'Predicted (Logistic Regression)': y_pred_lr_trimmed,
    'Predicted (KNN)': y_pred_knn_trimmed
})

print(pred_df.head(10))

# ---------------------------------------------------------
# User input prediction
# ---------------------------------------------------------
from sklearn.preprocessing import StandardScaler

# Ensure features are defined
features = x_train.columns.tolist()

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(x_train)  # Fit only on training data

print("\nEnter values for custom prediction:")
custom_input = []

# Collect user input for all features
for feature in features:
    value = float(input(f"Enter {feature}: "))
    custom_input.append(value)

# Convert input to numpy array and scale it
custom_input_scaled = scaler.transform([custom_input])

# Predictions from each model
custom_pred_xgb = xgb_model.predict(custom_input_scaled)
print(f"Predicted Temperature (XGBoost): {custom_pred_xgb[0]:.2f} \u00b0C")

custom_pred_rf = rf_model.predict(custom_input_scaled)
print(f"Predicted Temperature (Random Forest): {custom_pred_rf[0]:.2f} \u00b0C")

custom_pred_lr = log_reg.predict(custom_input_scaled)
print(f"Predicted Temperature (Logistic Regression): {custom_pred_lr[0]:.2f} \u00b0C")

custom_pred_knn = knn_model.predict(custom_input_scaled)
print(f"Predicted Temperature (KNN): {custom_pred_knn[0]:.2f} \u00b0C")
