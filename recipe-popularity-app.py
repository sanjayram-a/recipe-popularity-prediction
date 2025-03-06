# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the model
MODEL_PATH = 'recipe_popularity_model.pkl'
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'rating': float(request.form['rating']),
            'servings': int(request.form['servings']),
            'prep_time_minutes': int(request.form['prep_time_minutes']),
            'cook_time_minutes': int(request.form['cook_time_minutes']),
            'total_time_minutes': int(request.form['total_time_minutes']),
            'ingredient_count': int(request.form['ingredient_count']),
            'calories': float(request.form['calories']),
            'cuisine_type': request.form['cuisine_type']
        }
        
        # Create a DataFrame from the form data
        recipe_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(recipe_df)[0]
        
        return render_template('index.html', 
                               prediction=f"Predicted Popularity: {prediction:.2f}",
                               input_data=data)
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':    
    app.run(debug=True)
