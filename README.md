
# Revolutionizing Dairy Practices Using Machine Learning

This project leverages machine learning to optimize dairy farming practices by predicting suitable packaging types for dairy products. The solution includes an intuitive user interface, robust machine learning models, and a reliable backend to ensure accurate predictions and usability.

---

## Project Overview

### Key Features
- User-friendly UI for input fields like **Location**, **Total Land Area**, **Number of Cows**, **Farm Size**, **Product Name**, **Brand**, and **Shelf Life**.
- Prediction of suitable dairy packaging types, such as **Refrigerated** or **Tetra Pack**.
- Comparisons of model accuracies displayed dynamically.
- Deployment-ready structure for real-world use.

### Project Stack
1. **Frontend:** HTML, CSS, Bootstrap
2. **Backend:** Flask
3. **Machine Learning Models:**
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier
   - Random Forest Classifier
4. **Data Visualization:** Python (Matplotlib and Seaborn)
5. **Deployment:** Local server with Flask (Future: Cloud-based deployment)

---

## Process Workflow

### 1. **Data Collection and Preprocessing**
   - Collected data includes attributes like farm location, size, shelf life, and brand.
   - Preprocessed and cleaned data to ensure it is ready for modeling.

### 2. **Model Development**
   - Implemented KNN, Decision Tree, and Random Forest Classifiers.
   - Evaluated models using metrics like accuracy.
   - Selected Random Forest Classifier for deployment with an accuracy of **85.43%**.

### 3. **Pickle File Creation**
   - Saved the trained Random Forest Classifier model as `model.pkl` for efficient deployment and prediction.

### 4. **Frontend Development**
   - Designed input forms for user data.
   - Ensured the UI is simple and user-friendly for dairy farmers.

### 5. **Backend Development**
   - Used Flask for routing user inputs to the trained model.
   - Processed inputs to generate predictions dynamically and display results on the frontend.

---

## Results and Model Performance

### Model Predictions
| **Model**                  | **Prediction**   |
|----------------------------|------------------|
| KNN Classifier             | Refrigerated     |
| Decision Tree Classifier   | Refrigerated     |
| Random Forest Classifier   | Tetra Pack       |

### Model Accuracies
| **Model**                  | **Accuracy**     |
|----------------------------|------------------|
| KNN Classifier             | 67.17%           |
| Decision Tree Classifier   | 80.92%           |
| Random Forest Classifier   | 85.43%           |

---

## Steps to Use the Application

### 1. **Run the Flask App**
   - Clone the repository.
   - Install dependencies using `pip install -r requirements.txt`.
   - Start the app: `python app.py`.

### 2. **Input User Details**
   - Open the local server URL in your browser (e.g., `http://127.0.0.1:5000`).
   - Fill in the form with required inputs like location, farm size, and product details.

### 3. **View Predictions**
   - Submit the form to view packaging predictions.
   - Compare model accuracies on the results page.

### 4. **Pickle File**
   - The app uses `model.pkl` to load the pre-trained Random Forest Classifier.
   - Predictions are made in real-time without retraining the model.

---

## Screenshots of the Application

### 1. **Homepage**
   - Displays the user input form with fields like location, total land area, and shelf life.

### 2. **Results Page**
   - Displays predictions from each model and their respective accuracies.
   - Highlights the chosen packaging type based on the Random Forest Classifier.

### 3. **Model Comparison**
   - Side-by-side comparison of KNN, Decision Tree, and Random Forest results.

---

## Future Enhancements
- Integration with IoT devices for real-time data.
- Cloud deployment for wider accessibility.
- Additional features like milk quality analysis and pricing predictions.
