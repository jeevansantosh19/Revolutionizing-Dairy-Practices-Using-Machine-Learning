from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

application = Flask(__name__)

# Load models and accuracies
knn_model = joblib.load('models/knn_model.pkl')
dtc_model = joblib.load('models/dtc_model.pkl')
rfc_model = joblib.load('models/rfc_model.pkl')
model_accuracies = joblib.load('models/model_accuracies.pkl')

# Define storage condition mapping (using names directly)
storage_conditions = {
    'Refrigerated': 'Refrigerated',
    'Ambient': 'Ambient',
    'Frozen': 'Frozen',
    'Tetra Pack': 'Tetra Pack',
    'Polythene Packet': 'Polythene Packet'
}

# Initialize LabelEncoders
location_encoder = LabelEncoder()
location_encoder.fit(['Delhi', 'Chandigarh', 'Uttar Pradesh', 'Gujarat', 'Karnataka',
                      'Madhya Pradesh', 'Rajasthan', 'Maharashtra', 'Haryana', 'Kerala',
                      'Telangana', 'Jharkhand', 'Bihar', 'West Bengal', 'Tamil Nadu'])

farm_size_encoder = LabelEncoder()
farm_size_encoder.fit(['Small', 'Medium', 'Large'])

product_name_encoder = LabelEncoder()
product_name_encoder.fit(['Curd', 'Lassi', 'Paneer', 'Yogurt', 'Buttermilk',
                          'Butter', 'Milk', 'Ice Cream', 'Ghee', 'Cheese'])

brand_encoder = LabelEncoder()
brand_encoder.fit(['Amul', 'Mother Dairy', 'Raj', 'Sudha', 'Dodla Dairy',
                   'Palle2patnam', 'Dynamix Dairies', 'Warana',
                   'Parag Milk Foods', 'Passion Cheese', 'Britannia Industries'])

@application.route('/')
def home():
    return render_template('home.html')

@application.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data
    location = request.form['location']
    land_area = float(request.form['land_area'])
    num_cows = int(request.form['num_cows'])
    farm_size = request.form['farm_size']
    product_name = request.form['product_name']
    brand = request.form['brand']
    shelf_life = int(request.form['shelf_life'])

    # Encode categorical features
    location_encoded = location_encoder.transform([location])[0]
    farm_size_encoded = farm_size_encoder.transform([farm_size])[0]
    product_name_encoded = product_name_encoder.transform([product_name])[0]
    brand_encoded = brand_encoder.transform([brand])[0]

    # Prepare feature vector
    features = pd.DataFrame({
        'Location': [location_encoded],
        'Total Land Area (acres)': [land_area],
        'Number of Cows': [num_cows],
        'Farm Size': [farm_size_encoded],
        'Product Name': [product_name_encoded],
        'Brand': [brand_encoded],
        'Shelf Life (days)': [shelf_life]
    })

    # Predict using all models
    knn_output = knn_model.predict(features)[0]
    dtc_output = dtc_model.predict(features)[0]
    rfc_output = rfc_model.predict(features)[0]

    # Ensure predictions are valid (only valid storage condition values are allowed)
    predictions = {
        "KNN Classifier": storage_conditions.get(knn_output) if knn_output in storage_conditions else None,
        "Decision Tree Classifier": storage_conditions.get(dtc_output) if dtc_output in storage_conditions else None,
        "Random Forest Classifier": storage_conditions.get(rfc_output) if rfc_output in storage_conditions else None
    }

    # Remove invalid predictions (None) from the results
    valid_predictions = {model: prediction for model, prediction in predictions.items() if prediction is not None}

    # Retrieve accuracies
    knn_accuracy = model_accuracies["KNN Classifier"]
    dtc_accuracy = model_accuracies["Decision Tree Classifier"]
    rfc_accuracy = model_accuracies["Random Forest Classifier"]

    accuracies = {
        "KNN Classifier": f"{knn_accuracy * 100:.2f}%",
        "Decision Tree Classifier": f"{dtc_accuracy * 100:.2f}%",
        "Random Forest Classifier": f"{rfc_accuracy * 100:.2f}%"
    }

    # Render template with only predictions and accuracies
    return render_template('results.html', predictions=valid_predictions, accuracies=accuracies)

if __name__ == '__main__':
    application.run(debug=True)