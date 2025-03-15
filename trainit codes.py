import pandas as pd

try:
    df = pd.read_csv('data.csv')
    print(df.info())
    display(df.head())
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please make sure the file exists in the current directory.")
    df = None # Assign None to df to indicate failure
except pd.errors.EmptyDataError:
    print("Error: 'data.csv' is empty.")
    df = None
except pd.errors.ParserError:
    print("Error: Unable to parse 'data.csv'. Please check the file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None
    import pandas as pd

try:
    df = pd.read_csv('data.csv', encoding='latin-1')
    print(df.info())
    display(df.head())
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please make sure the file exists in the current directory.")
    df = None
except pd.errors.EmptyDataError:
    print("Error: 'data.csv' is empty.")
    df = None
except pd.errors.ParserError:
    print("Error: Unable to parse 'data.csv'. Please check the file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None
    import matplotlib.pyplot as plt
import seaborn as sns

# 1. Examine Data Structure
print("Data Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)

# 2. Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# Visualize missing values (Heatmap)
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# 3. Descriptive Statistics (Numerical Features)
numerical_cols = df.select_dtypes(include=['number']).columns
print("\nDescriptive Statistics for Numerical Features:\n", df[numerical_cols].describe())


# 4. & 5. Data Distribution Visualization (Histograms, Box Plots, Target Variable Analysis)
plt.figure(figsize=(15, 10))

# Histograms for numerical features
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Box plots for numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
  plt.subplot(3, 3, i + 1)
  sns.boxplot(y=df[col])
  plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# Specific analysis for AQI (Assuming 'pm2_5' is the AQI here)
if 'pm2_5' in numerical_cols:
    print("\nDescriptive Statistics for pm2_5 (AQI):\n", df['pm2_5'].describe())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['pm2_5'], kde=True)
    plt.title('Distribution of pm2_5 (AQI)')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['pm2_5'])
    plt.title('Box Plot of pm2_5 (AQI)')

    plt.tight_layout()
    plt.show()
else:
    print("pm2_5 column not found in the DataFrame.")
import pandas as pd

# 1. Handle Missing Values
# Impute numerical features with the median
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    if col != 'pm2_5':  # Exclude 'pm2_5' as instructed
        df[col] = df[col].fillna(df[col].median())

# Fill missing 'location_monitoring_station' with the mode
df['location_monitoring_station'] = df['location_monitoring_station'].fillna(df['location_monitoring_station'].mode()[0])

# Remove rows with missing 'pm2_5' values
df = df.dropna(subset=['pm2_5'])


# 2. Outlier Handling (IQR method)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# 3. Data Type Conversion
df['sampling_date'] = pd.to_datetime(df['sampling_date'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 4. Duplicate Removal
df = df.drop_duplicates()

display(df.head())
import pandas as pd

# 1. Handle Missing Values
# Impute numerical features with the median
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    if col != 'pm2_5':  # Exclude 'pm2_5' as instructed
        df.loc[:, col] = df[col].fillna(df[col].median())

# Fill missing 'location_monitoring_station' with the mode
df.loc[:, 'location_monitoring_station'] = df['location_monitoring_station'].fillna(df['location_monitoring_station'].mode()[0])

# Remove rows with missing 'pm2_5' values
df = df.dropna(subset=['pm2_5'])


# 2. Outlier Handling (IQR method)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df.loc[:, col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# 3. Data Type Conversion
# Attempt to infer the format from the first few valid entries
first_valid_date_index = df['sampling_date'].first_valid_index()
if first_valid_date_index is not None:
    first_valid_date = df.loc[first_valid_date_index, 'sampling_date']

    try:
        pd.to_datetime(first_valid_date)
        df['sampling_date'] = pd.to_datetime(df['sampling_date'], errors='coerce')
    except Exception as e:
        print(f"Error converting sampling_date: {e}")

df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 4. Duplicate Removal
df = df.drop_duplicates()

display(df.head())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Interaction Terms
df['temp_pm25'] = df['pm2_5'] * df['so2']
df['humidity_pm10'] = df['pm2_5'] * df['no2']

# 2. Polynomial Features
df['temp_squared'] = df['so2']**2
df['humidity_cubed'] = df['no2']**3


# 3. Day of the Week Feature
df['sampling_date'] = pd.to_datetime(df['sampling_date'])
df['day_of_week'] = df['sampling_date'].dt.dayofweek

# 4. Evaluate New Features
# Correlation Analysis
correlation_matrix = df[['pm2_5', 'temp_pm25', 'humidity_pm10', 'temp_squared', 'humidity_cubed', 'day_of_week']].corr()
display(correlation_matrix)

# Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of New Features')
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(['temp_pm25', 'humidity_pm10', 'temp_squared', 'humidity_cubed', 'day_of_week']):
    plt.subplot(3, 2, i + 1)
    sns.scatterplot(x=df[col], y=df['pm2_5'])
    plt.title(f'{col} vs pm2_5')

plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = df.drop('pm2_5', axis=1)
y = df['pm2_5']

# Convert 'sampling_date' and 'date' columns to numerical representations if they exist
if 'sampling_date' in X.columns:
    X['sampling_date'] = pd.to_numeric(pd.to_datetime(X['sampling_date']))
if 'date' in X.columns:
    X['date'] = pd.to_numeric(pd.to_datetime(X['date']))


# Convert object columns to category type
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes

# First split: 70% train, 30% temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=pd.cut(y, bins=10)
)

# Second split: 50% val, 50% test from temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=pd.cut(y_temp, bins=10)
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Instantiate the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regression': SVR()
}

# Train the models and evaluate their performance
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    results[model_name] = mse

# Print the results
for model_name, mse in results.items():
    print(f"{model_name}: MSE = {mse}")
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1  # Use all available cores
)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Get the best estimator
best_rf_model = grid_search.best_estimator_
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Discretize the AQI values into 5 bins
bins = [0, 50, 100, 200, 300, 500]  # Define AQI bins
labels = [1, 2, 3, 4, 5] # Define labels for each bin
y_test_disc = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)
y_pred_disc = pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True)

# Calculate the confusion matrix
cm = confusion_matrix(y_test_disc, y_pred_disc)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test_disc, y_pred_disc)
precision = precision_score(y_test_disc, y_pred_disc, average='weighted')
recall = recall_score(y_test_disc, y_pred_disc, average='weighted')
f1 = f1_score(y_test_disc, y_pred_disc, average='weighted')

# Print the results
print("Confusion Matrix:\n", cm)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
