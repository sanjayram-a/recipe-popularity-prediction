import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import joblib

# Load the dataset
df = pd.read_csv("recipes_with_popularity.csv")  # Replace with your actual file path

# Basic data exploration
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Function to convert time strings to minutes
def extract_minutes(time_str):
    if pd.isna(time_str):
        return np.nan
    
    hours = 0
    minutes = 0
    
    # Extract hours if present
    hours_match = re.search(r'(\d+)\s*hour', time_str, re.IGNORECASE)
    if hours_match:
        hours = int(hours_match.group(1))
    
    # Extract minutes if present
    minutes_match = re.search(r'(\d+)\s*minute', time_str, re.IGNORECASE)
    if minutes_match:
        minutes = int(minutes_match.group(1))
    
    return hours * 60 + minutes

# Apply time conversion
df['prep_time_minutes'] = df['prep_time'].apply(extract_minutes)
df['cook_time_minutes'] = df['cook_time'].apply(extract_minutes)
df['total_time_minutes'] = df['total_time'].apply(extract_minutes)

# Extract number of ingredients
df['ingredient_count'] = df['ingredients'].apply(lambda x: len(str(x).split(',')))

# Extract cuisine type from cuisine_path
df['cuisine_type'] = df['cuisine_path'].apply(lambda x: str(x).split('/')[-1] if pd.notna(x) else 'unknown')

# Extract nutrition info - calories, fat, protein, and carbs
def extract_nutrition_value(nutrition_str, nutrient):
    if pd.isna(nutrition_str):
        return np.nan
    
    pattern = r'{}:\s*([\d.]+)'.format(nutrient)
    match = re.search(pattern, nutrition_str, re.IGNORECASE)
    
    if match:
        return float(match.group(1))
    return np.nan

df['calories'] = df['nutrition'].apply(lambda x: extract_nutrition_value(x, 'calories'))
df['fat'] = df['nutrition'].apply(lambda x: extract_nutrition_value(x, 'fat'))
df['protein'] = df['nutrition'].apply(lambda x: extract_nutrition_value(x, 'protein'))
df['carbs'] = df['nutrition'].apply(lambda x: extract_nutrition_value(x, 'carbohydrates'))

# Fill missing values in popularity (target variable)
df['popularity'].fillna(df['popularity'].median(), inplace=True)

# Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'])
plt.title('Distribution of Recipe Popularity')
plt.xlabel('Popularity')
plt.savefig('popularity_distribution.png')
plt.close()

# Correlation between numerical features and popularity
numerical_features = ['servings', 'rating', 'prep_time_minutes', 'cook_time_minutes', 
                      'total_time_minutes', 'ingredient_count', 'calories', 'fat', 
                      'protein', 'carbs']

# Filter only numeric columns that have data
numeric_df = df[numerical_features].copy()
numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

# Calculate correlation with popularity
correlations = numeric_df.corrwith(df['popularity']).sort_values(ascending=False)
print("\nCorrelations with popularity:")
print(correlations)

# Create a correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_heatmap.png')
plt.close()

# Feature selection
selected_features = [
    'rating', 
    'servings',
    'prep_time_minutes',
    'cook_time_minutes',
    'total_time_minutes',
    'ingredient_count',
    'calories',
    'cuisine_type'
]

# Drop rows with missing values in selected features
X = df[selected_features].copy()
y = df['popularity']

# Define preprocessing for numerical and categorical features
numerical_features = ['rating', 'servings', 'prep_time_minutes', 'cook_time_minutes', 
                      'total_time_minutes', 'ingredient_count', 'calories']
categorical_features = ['cuisine_type']

# Create transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the modeling pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared:", r2_score(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print("\nCross-Validation MAE:", -cv_scores.mean())

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model_pipeline, 
    param_grid, 
    cv=3, 
    scoring='neg_mean_absolute_error',
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", -grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test)

print("\nBest Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, best_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, best_predictions)))
print("R-squared:", r2_score(y_test, best_predictions))

# Feature importance
best_rf = best_model.named_steps['model']
feature_names = (
    numerical_features +
    list(best_model.named_steps['preprocessor']
         .named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Save the best model
joblib.dump(best_model, 'recipe_popularity_model.pkl')

print("\nModel saved as 'recipe_popularity_model.pkl'")

# Simple prediction function for new recipes
def predict_popularity(recipe_data):
    """
    Predict the popularity of a new recipe.
    
    Parameters:
    recipe_data (dict): A dictionary containing recipe features
    
    Returns:
    float: Predicted popularity score
    """
    model = joblib.load('recipe_popularity_model.pkl')
    recipe_df = pd.DataFrame([recipe_data])
    return model.predict(recipe_df)[0]

# Example usage:
example_recipe = {
    'rating': 4.5,
    'servings': 4,
    'prep_time_minutes': 30,
    'cook_time_minutes': 45,
    'total_time_minutes': 75,
    'ingredient_count': 12,
    'calories': 350,
    'cuisine_type': 'italian'
}

print("\nExample prediction for a new recipe:")
print(f"Predicted popularity: {predict_popularity(example_recipe):.2f}")
