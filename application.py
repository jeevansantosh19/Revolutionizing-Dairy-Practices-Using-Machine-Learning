# Importing the Required Libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import flask
from flask import Flask
from flask import render_template, request, url_for
import io
import base64
import warnings
warnings.filterwarnings("ignore")

# Uploading a Dataset
diary_data = pd.read_csv("dairy_dataset.csv")
diary_data

# Displaying a First Rows of a Dataset
diary_data.head(1500)

# Displaying a Last Rows of a Dataset
diary_data.tail(1500)

# Displaying Columns of a Dataset
diary_data.columns

# Displaying a Datatypes of Dataset
diary_data.dtypes

# Checking a Null values in a dataset
diary_data.isnull().sum()

# Dropping a Unnecessary Columns
columns = ['Date', 'Product ID', 'Quantity (liters/kg)', 'Price per Unit', 'Total Value', 'Production Date', 'Expiration Date', 'Quantity Sold (liters/kg)', 'Price per Unit (sold)', 'Approx. Total Revenue(INR)', 'Customer Location', 'Quantity in Stock (liters/kg)', 'Minimum Stock Threshold (liters/kg)', 'Reorder Quantity (liters/kg)']
columns

latest_data = diary_data.drop(columns, axis = 1)
latest_data

labels = ['Location', 'Farm Size', 'Product Name', 'Brand', 'Storage Condition', 'Sales Channel']
labels

# Displaying a First Rows from Latest Dataset
latest_data.head(500)

# Displaying a Last Rows from latest Dataset
latest_data.tail(500)

# Creating Other Dataset
dataset1 = latest_data.copy()
dataset1

# Initializing a Label Encoder
label_encoder = LabelEncoder()
label_encoder

# Encoding a Categorical Columns
dataset1['Location'] = label_encoder.fit_transform(latest_data['Location'])
dataset1['Farm Size'] = label_encoder.fit_transform(latest_data['Farm Size'])
dataset1['Product Name'] = label_encoder.fit_transform(latest_data['Product Name'])
dataset1['Brand'] = label_encoder.fit_transform(latest_data['Brand'])
dataset1['Storage Condition'] = label_encoder.fit_transform(latest_data['Storage Condition'])
dataset1['Sales Channel'] = label_encoder.fit_transform(latest_data['Sales Channel'])
dataset1

# Initializing a Standard Scaler
standard_scaler = StandardScaler()
standard_scaler

# Scaling the Numeric Columns
numeric_columns = ['Total Land Area (acres)', 'Number of Cows', 'Shelf Life (days)']
numeric_columns

# Standardizing and Transforming the Dataset
dataset1[numeric_columns] = standard_scaler.fit_transform(dataset1[numeric_columns])
dataset1

# Feature Varaible
X = dataset1.drop('Storage Condition', axis = 1)
X

# Target Variable
y = dataset1['Storage Condition']
y

# Dividing into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("X_train shape:", X_train.shape)
print("X_train columns:", X_train.columns)
print("X_test shape:", X_test.shape)
print("X_test columns:", X_test.columns)
print("y_train shape:", y_train.shape)
print("y_test shape:",y_test.shape)

column1 = ['Location', 'Farm Size', 'Product Name']

train_data = X_train.drop(column1, axis=1)
train_data

test_data = X_test.drop(column1, axis=1)
test_data

train_data = X_train
test_data = X_test
train_target = y_train
test_target = y_test

# Defining the function to encode plots to Base64
def encode_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0) 
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close() 
    return encoded

# Defining the function to format the classification report
def format_classification_report(report):
    lines = report.split("\n")
    formatted = []
    for line in lines:
        if line.strip(): 
            formatted.append(line)
    return "<br>".join(formatted)

# Building the KNN Classifier
knnc = KNeighborsClassifier(n_neighbors = 3, metric='euclidean')
knnc

knnc.fit(train_data, train_target)

# Predicting for KNN Classifier
y_pred_knn = knnc.predict(test_data)
y_pred_knn

# Initializing the Train Scores and Test Scores
print("Train and Test Scores of KNN Classifier:")
train_scores_knn = []
test_scores_knn = []

for i in range(3,27,2):

    knnc = KNeighborsClassifier(n_neighbors=i)
    knnc.fit(train_data, train_target)
    train_scores_knn.append(knnc.score(train_data, train_target))
    test_scores_knn.append(knnc.score(test_data, test_target))
    
print("Train Scores:",train_scores_knn)
print("Test Scores:",test_scores_knn)

round_train_scores_knn = [round(num,2) for num in train_scores_knn]
print("Train scores (Rounded):",round_train_scores_knn)
round_test_scores_knn = [round(num,2) for num in test_scores_knn]
print("Test scores (Rounded):",round_test_scores_knn)

# Visualizations for Both Training and Testing Data
plt.figure(figsize=(10,6))
plt.plot(range(3,27,2), round_train_scores_knn)
plt.title('Train Scores vs. K Value')
plt.xlabel('K')
plt.ylabel('Train Scores')
plt.legend()
train_scores_knn_img = encode_plot_to_base64()
plt.close()

plt.figure(figsize=(10,6))
plt.plot(range(3,27,2), round_test_scores_knn)
plt.title('Test Scores vs. K Value')
plt.xlabel('K')
plt.ylabel('Test Scores')
plt.legend()
test_scores_knn_img = encode_plot_to_base64()
plt.close()

# Initializing the Error Rate
print("Error Rate of KNN Classifier:")
error_rate_knn = []
for i in range(3,27,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_data,train_target)
    y_pred_knn = knn.predict(X_test)
    error_rate_knn.append(np.mean(y_pred_knn  != y_test))
    
print("Error Rates:",error_rate_knn)

round_error_rate_knn = [round(num,2) for num in error_rate_knn]
print("Error Rate: ", round_error_rate_knn)

# Visualization for Error Rate
plt.figure(figsize=(10,6))
plt.plot(range(3,27,2),round_error_rate_knn)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.legend()
error_rate_knn_img = encode_plot_to_base64()
plt.close()

# Evaluating the Metrics of KNN Classifier
print("\nK Nearest Neighbors Classifier")
accuracy_knn = accuracy_score(test_target, y_pred_knn)
print("\nAccuracy:", round(accuracy_knn*100),"%")

print("\nClassification Report:")
class_rprt_knn = classification_report(y_pred_knn, test_target)
print(class_rprt_knn)

formatted_report_knn = format_classification_report(class_rprt_knn)

print("\nConfusion Matrix:")
conf_matrix_knn = confusion_matrix(y_pred_knn, test_target)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
confusion_matrix_knn_img = encode_plot_to_base64()
plt.close()

# Building the Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion = 'gini', random_state = 42, max_depth = 3)
dtc

# Training the DTC Model
dtc.fit(train_data, train_target)

# Predicting for DecisionTreeClassifier
y_pred_dtc = dtc.predict(test_data)
y_pred_dtc

# Initializing the Train and Test Scores
print("Train and Test Scores of Decision Tree Classifier:")
train_scores_dtc = []
test_scores_dtc = []

for i in range(1, 21):  
    dtc = DecisionTreeClassifier(max_depth=i)
    dtc.fit(train_data, train_target)
    train_scores_dtc.append(dtc.score(train_data, train_target))
    test_scores_dtc.append(dtc.score(test_data, test_target))

print("Train Scores:", train_scores_dtc)
print("Test Scores:", test_scores_dtc)

round_train_scores_dtc = [round(num, 2) for num in train_scores_dtc]
print("Train Scores (Rounded):", round_train_scores_dtc)

round_test_scores_dtc = [round(num, 2) for num in test_scores_dtc]
print("Test Scores (Rounded):", round_test_scores_dtc)

# Initializing the Error Rate
print("Error Rate of Decision Tree Classifier:")
error_rates_dtc = []
for i in range(1, 21): 
    dtc = DecisionTreeClassifier(max_depth=i, random_state=42)
    dtc.fit(train_data, train_target)
    y_pred_dtc = dtc.predict(test_data)
    error_rates_dtc.append(np.mean(y_pred_dtc != test_target))

print("Error Rate",error_rates_dtc)

round_error_rates_dtc = [round(num, 2) for num in error_rates_dtc]
print("Error Rates (Rounded):", round_error_rates_dtc)

# Visualizations for Both Training and Testing Data
plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_train_scores_dtc, marker='o', label='Train Scores', color='blue', linestyle='--')
plt.title('Performance Metrics vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Performance Metrics')
plt.legend()
train_scores_dtc_img = encode_plot_to_base64()
plt.close()

plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_test_scores_dtc, marker='s', label='Test Scores', color='orange', linestyle='dotted')
plt.title('Performance Metrics vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Performance Metrics')
plt.legend()
test_scores_dtc_img = encode_plot_to_base64()
plt.close()

# Visualization for Error Rate
plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_error_rates_dtc, marker='o', label='Error Rates', color='red', linestyle='solid')
plt.title('Performance Metrics vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Performance Metrics')
plt.legend()
error_rate_dtc_img = encode_plot_to_base64()
plt.close()

# Evaluating the Metrics of DTC Classifier
print("\nDecision Tree Classifier")
accuracy_dtc = accuracy_score(test_target, y_pred_dtc)
print("\nAccuracy:", round(accuracy_dtc*100),"%")

print("\nClassification Report:")
class_rprt_dtc = classification_report(y_pred_dtc ,test_target)
print(class_rprt_dtc)

formatted_report_dtc = format_classification_report(class_rprt_dtc)

print("\nConfusion Matrix:")
conf_matrix_dtc = confusion_matrix(y_pred_dtc ,test_target)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_dtc, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
confusion_matrix_dtc_img = encode_plot_to_base64()
plt.close()

# Building the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
rfc

# Training the Model
rfc.fit(train_data, train_target)

# Predicting on Random Forest Classifier
y_pred_rfc = rfc.predict(test_data)
y_pred_rfc

# Initializing the Train and Test Scores
print("Train and Test Scores of Random Forest Classifier:")
train_scores_rfc = []
test_scores_rfc = []

for i in range(1, 21):
    rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=i, random_state=42)
    rfc.fit(train_data, train_target)
    train_scores_rfc.append(rfc.score(train_data, train_target))
    test_scores_rfc.append(rfc.score(test_data, test_target))

print("Train Scores:", train_scores_rfc)
print("Test Scores:", test_scores_rfc)

round_train_scores_rfc = [round(num, 2) for num in train_scores_rfc]
print("Train Scores (Rounded):", round_train_scores_rfc)

round_test_scores_rfc = [round(num, 2) for num in test_scores_rfc]
print("Test Scores (Rounded):", round_test_scores_rfc)

# Initializing the Error Rate
print("Error Rate of Random Forest Classifier:")
error_rate_rfc = []
for i in range(1, 21):  
    rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=i, random_state=42)
    rfc.fit(train_data, train_target)
    y_pred_rfc = rfc.predict(test_data)
    error_rate_rfc.append(np.mean(y_pred_rfc != test_target))

print("Error Rate:",error_rate_rfc)

round_error_rate_rfc = [round(rate, 2) for rate in error_rate_rfc]
print("Error Rates (Rounded):", round_error_rate_rfc)

# Visualizing the Train and Test Scores
plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_train_scores_rfc, marker='o', label='Train Scores', color='blue', linestyle='--')
plt.title('Performance Metrics vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Performance Metrics')
plt.legend()
train_scores_rfc_img = encode_plot_to_base64()
plt.close()

plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_test_scores_rfc, marker='s', label='Test Scores', color='orange', linestyle='solid')
plt.title('Performance Metrics vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Performance Metrics')
plt.legend()
test_scores_rfc_img = encode_plot_to_base64()
plt.close()

# Visualization for Error Rate
plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_error_rate_rfc, marker='^', label='Error Rates', color='red', linestyle='dotted')
plt.title('Performance Metrics vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Performance Metrics')
plt.legend()
error_rate_rfc_img = encode_plot_to_base64()
plt.close()

# Evaluating the Metrics of Random Forest Classifier
print("\nRandom Forest Classifier")
accuracy_rfc = accuracy_score(test_target, y_pred_rfc)
print("\nAccuracy:", round(accuracy_rfc*100),"%")

print("\nClassification Report:")
class_rprt_rfc = classification_report(y_pred_rfc ,test_target)
print(class_rprt_rfc)

formatted_report_rfc = format_classification_report(class_rprt_rfc)

print("\nConfusion Matrix:")
conf_matrix_rfc = confusion_matrix(y_pred_rfc ,test_target)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_rfc, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
confusion_matrix_rfc_img = encode_plot_to_base64()
plt.close()

# Building the Support Vector Machine Classifier
svm = SVC(C = 1.0, kernel = 'linear', gamma = 'scale', random_state = 42)
svm

# Training the Model
svm.fit(train_data, train_target)

# Predicting on SVM
y_pred_svm = svm.predict(test_data)
y_pred_svm

# Initializing the Train and Test Scores
print("Train and Test Scores of Support Vector Machine Classifier:")
train_scores_svm = []
test_scores_svm = []

for i in range(1, 21):
    svm = SVC(C = i, kernel = 'linear', gamma = 'scale', random_state = 42)
    svm.fit(train_data, train_target)
    train_scores_svm.append(svm.score(train_data, train_target))
    test_scores_svm.append(svm.score(test_data, test_target))

print("Train Scores:",train_scores_svm)
print("Test Scores:",test_scores_svm)

round_train_scores_svm = [round(num, 2) for num in train_scores_svm]
print("Train Scores (Rounded):",round_train_scores_svm)

round_test_scores_svm = [round(num, 2) for num in test_scores_svm]
print("Test Scores (Rounded):",round_test_scores_svm)

# Initializing the Error Rate
print("Error Rate of Support Vector Machine Classifier:")
error_rate_svm = []

for i in range(1, 21):
    svm = SVC(C = i, kernel = 'linear', gamma = 'scale', random_state = 42)
    svm.fit(train_data, train_target)
    y_pred_svm = svm.predict(test_data)
    error_rate_svm.append(np.mean(y_pred_svm != test_target))

print("Error Rate:",error_rate_svm)

round_error_rate_svm = [round(num, 2) for num in error_rate_svm]
print("Error Rate (Rounded):",round_error_rate_svm)

# Visualizing the Train and Test Scores
plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_train_scores_svm, marker='o', label='Train Scores', color='blue', linestyle='--')
plt.title("Training Scores vs C (SVC with Linear Kernel)")
plt.xlabel("Training Scores")
plt.ylabel("C (Regularization Parameter)")
plt.legend()
train_scores_svm_img = encode_plot_to_base64()
plt.close()

plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_test_scores_svm, marker='o', label='Test Scores', color='red', linestyle='solid')
plt.title("Testing Scores vs C (SVC with Linear Kernel)")
plt.xlabel("Testing Scores")
plt.ylabel("C (Regularization Parameter)")
plt.legend()
test_scores_svm_img = encode_plot_to_base64()
plt.close()

# Visualizing the Error Rate
plt.figure(figsize=(10,6))
plt.plot(range(1, 21))
plt.plot(range(1, 21), round_error_rate_svm, marker='o', label='Error Rate', color='green', linestyle='dotted')
plt.title("Error Rate vs C (SVC with Linear Kernel)")
plt.xlabel("Error Rate")
plt.ylabel("C (Regularization Parameter)")
plt.legend()
error_rate_svm_img = encode_plot_to_base64()
plt.close()

# Evaluating the Metrics of Support Vector Machine Classifier
print("\nSupport Vector Machine Classifier")
accuracy_svm = accuracy_score(test_target, y_pred_svm)
print("\nAccuracy:", round(accuracy_svm*100),"%")

print("\nClassification Report:")
class_rprt_svm = classification_report(y_pred_svm, test_target)
print(class_rprt_svm)

formatted_report_svm = format_classification_report(class_rprt_svm)

print("\nConfusion Matrix:")
conf_matrix_svm = confusion_matrix(y_pred_svm ,test_target)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
confusion_matrix_svm_img = encode_plot_to_base64()
plt.close()

# Deployment
# Creating an Instance of a Flask
application = Flask(__name__)

# Opening a Home Page
@application.route('/')
def main():
    return render_template('home.html')

# Navigating to Results after clicking predict button
@application.route('/Predict', methods = ['POST'])
def predict():
    location = request.form.get('location')
    total_land_area = request.form.get('total_land_area')
    no_of_cows = request.form.get('no_of_cows')
    farm_size = request.form.get('farm_size')
    product_name = request.form.get('product_name')
    brand = request.form.get('brand')
    shelf_life = request.form.get('shelf_life')
    storage_condition = request.form.get('storage_condition')
    sales_channel = request.form.get('sales_channel')

    # User Input
    user_input = {
        'Location: ': location,
        'Total Land Area (in Acres): ':total_land_area,
        'Number of Cows: ':no_of_cows,
        'Farm Size: ':farm_size,
        'Product Name: ':product_name,
        'Brand: ':brand,
        'Shelf Life (in days): ':shelf_life,
        'Storage Condition: ':storage_condition,
        'Sales Channel: ':sales_channel
    }
    
    return render_template('results.html', user_input = user_input)

# Navigating from Results to K Nearest Neighbors Classifier
@application.route('/knnc')
def knnc():
    global train_scores_knn_img, test_scores_knn_img, error_rate_knn_img, confusion_matrix_knn_img
    return render_template(
        'knnc.html',
        round_train_scores_knn=round_train_scores_knn,
        round_test_scores_knn=round_test_scores_knn,
        round_error_rate_knn=round_error_rate_knn,
        train_scores_knn_img=train_scores_knn_img,
        test_scores_knn_img=test_scores_knn_img,
        error_rate_knn_img=error_rate_knn_img,
        confusion_matrix_knn_img=confusion_matrix_knn_img,
        accuracy = accuracy_knn,
        formatted_report_knn = formatted_report_knn
    )

# Navigating from Results to Decision Tree Classifier
@application.route('/dtc')
def dtc():
    global train_scores_dtc_img, test_scores_dtc_img, error_rate_dtc_img, confusion_matrix_dtc_img
    return render_template(
        'dtc.html',
        round_train_scores_dtc = round_train_scores_dtc,
        round_test_scores_dtc = round_test_scores_dtc,
        round_error_rates_dtc = round_error_rates_dtc,
        train_scores_dtc_img = train_scores_dtc_img,
        test_scores_dtc_img = test_scores_dtc_img,
        error_rate_dtc_img = error_rate_dtc_img,
        confusion_matrix_dtc_img = confusion_matrix_dtc_img,
        accuracy = accuracy_dtc,
        formatted_report_dtc = formatted_report_dtc,
    )

# Navigating from Results to Random Forest Classifier
@application.route('/rfc')
def rfc():
    global train_scores_rfc_img, test_scores_rfc_img, error_rate_rfc_img, confusion_matrix_rfc_img
    return render_template(
        'rfc.html',
        round_train_scores_rfc = round_train_scores_rfc,
        round_test_scores_rfc = round_test_scores_rfc,
        round_error_rate_rfc = round_error_rate_rfc,
        train_scores_rfc_img = train_scores_rfc_img,
        test_scores_rfc_img = test_scores_rfc_img,
        error_rate_rfc_img = error_rate_rfc_img,
        confusion_matrix_rfc_img = confusion_matrix_rfc_img,
        accuracy = accuracy_rfc,
        formatted_report_rfc = formatted_report_rfc
    )

# Navigating from Results to Support Vector Machine Classifier
@application.route('/svm')
def svm():
    global train_scores_svm_img, test_scores_svm_img, error_rate_svm_img, confusion_matrix_svm_img
    return render_template(
        'svm.html',
        round_train_scores_svm = round_train_scores_svm,
        round_test_scores_svm = round_test_scores_svm,
        round_error_rate_svm = round_error_rate_svm,
        train_scores_svm_img = train_scores_svm_img, 
        test_scores_svm_img = test_scores_svm_img,
        error_rate_svm_img = error_rate_svm_img,
        confusion_matrix_svm_img = confusion_matrix_svm_img,
        accuracy = accuracy_svm,
        formatted_report_svm = formatted_report_svm
    )

# Driver Code
if __name__ == "__main__":
    application.run()