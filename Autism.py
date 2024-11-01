# Autism Spectrum Disorder Detection using Machine Learning (Enhanced with Visualizations)
# This script preprocesses the dataset, trains multiple ML models, evaluates their performance,
# and includes feature importance, correlation heatmap, ROC curves, and confusion matrix heatmaps.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"D:\Frnds Projects\Autism\Toddler Autism dataset July 2023.csv"  # Using a raw string for file path
data = pd.read_csv(file_path)

# Drop 'Case_No' column as it's only an identifier
data = data.drop(columns=['Case_No'])

# Convert target column to binary (1 for 'Yes', 0 for 'No')
data['Traits '] = data['Traits '].apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)

# Separate features and target
X = data.drop('Traits ', axis=1)
y = data['Traits ']

# Encode categorical features
categorical_features = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test']
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Dictionary to store metrics and ROC values for visualization
conf_matrices = {}
roc_values = {}

for model_name, model in classifiers.items():
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Store confusion matrix
    conf_matrices[model_name] = confusion_matrix(y_test, y_pred)
    
    # Print performance metrics
    print(f"{model_name} Performance Metrics")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(conf_matrices[model_name])
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")

    # Store ROC values
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_values[model_name] = (fpr, tpr, roc_auc)

# Correlation Heatmap with numeric columns only
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Confusion Matrix Heatmaps for each model
fig, axes = plt.subplots(1, len(classifiers), figsize=(20, 5))
for ax, (model_name, cm) in zip(axes, conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curves for each model
plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr, roc_auc) in roc_values.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.show()

# Feature Importance Plot for Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
feature_importance = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, palette="viridis")
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Final Model Accuracy Comparison
model_accuracies = [accuracy_score(y_test, model.predict(X_test)) for model in classifiers.values()]
plt.figure(figsize=(10, 6))
sns.barplot(x=list(classifiers.keys()), y=model_accuracies, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.05)
plt.show()
