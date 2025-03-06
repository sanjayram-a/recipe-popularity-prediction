# Recipe Popularity Prediction

This project provides a machine learning model to predict the popularity of recipes based on their characteristics. It includes a Flask web application for user interaction and a machine learning pipeline for training and evaluating the prediction model.

## Features
- **Predict Recipe Popularity**: Enter recipe details to get a popularity score.
- **Flask Web Application**: User-friendly interface for making predictions.
- **Machine Learning Pipeline**: Data preprocessing, feature engineering, model training, and evaluation.
- **Hyperparameter Tuning**: Uses GridSearchCV for optimizing the RandomForestRegressor model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recipe-popularity-prediction.git
   cd recipe-popularity-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web Application

1. Train the model and generate `recipe_popularity_model.pkl`:
   ```bash
   python recipe-popularity-prediction.py
   ```

2. Start the Flask app:
   ```bash
   python recipe-popularity-app.py
   ```

3. Open your browser and go to `http://127.0.0.1:5000/`

## Project Structure
```
recipe-popularity-prediction/
│── recipe-popularity-app.py        # Flask web application
│── recipe-popularity-prediction.py # Machine learning pipeline
│── recipes.csv                     # Dataset (replace with actual file)
│── recipe_popularity_model.pkl     # Trained model (generated after training)
│── templates/
│   ├── index.html                   # HTML template for web app
│── static/
│   ├── images/                      # Visualizations
│── README.md                        # Documentation
```

## Model Details
The model is trained using a **Random Forest Regressor**, with the following features:
- `rating`
- `servings`
- `prep_time_minutes`
- `cook_time_minutes`
- `total_time_minutes`
- `ingredient_count`
- `calories`
- `cuisine_type`

### Performance Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared Score (R²)**

## Example API Request
You can send a `POST` request to the `/predict` endpoint with the following data:
```json
{
    "rating": 4.5,
    "servings": 4,
    "prep_time_minutes": 30,
    "cook_time_minutes": 45,
    "total_time_minutes": 75,
    "ingredient_count": 12,
    "calories": 350,
    "cuisine_type": "italian"
}
```

## License
This project is licensed under the MIT License.

## Author
Your Name - [GitHub Profile](https://github.com/sanjayram-a)

