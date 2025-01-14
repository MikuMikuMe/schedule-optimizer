# Schedule-Optimizer

Creating a comprehensive schedule optimizer using machine learning involves several steps, including data preparation, model training, prediction, and evaluation. We'll build a simplified version of such a tool in Python that focuses on these core components. For a complete real-world system, additional considerations such as integration with existing HR systems, more sophisticated models, and real-world constraints would be necessary.

Below is a simplified Python program that demonstrates a schedule optimizer using machine learning. We'll use the `scikit-learn` library for model training and prediction, and include example data and comments for clarity.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data: Replace this with a dataset specific to employee scheduling
data = pd.DataFrame({
    'EmployeeID': range(1, 21),
    'CurrentShift': np.random.choice(['Morning', 'Afternoon', 'Night'], 20),
    'PreferredShift': np.random.choice(['Morning', 'Afternoon', 'Night'], 20),
    'WorkHours': np.random.randint(20, 40, 20),  # Weekly work hours
    'Performance': np.random.choice(['High', 'Average', 'Low'], 20)
})

# Encoding categorical features
data_encoded = pd.get_dummies(data, columns=['CurrentShift', 'PreferredShift', 'Performance'])

# Assume the target variable is employee satisfaction represented by preferred shift matching
data_encoded['Satisfaction'] = (data['PreferredShift'] == data['CurrentShift']).astype(int)

# Define features and target
features = data_encoded.drop(columns=['EmployeeID', 'Satisfaction'])
target = data_encoded['Satisfaction']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)

try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")

# Making predictions
try:
    predictions = model.predict(X_test)
except Exception as e:
    print(f"Error during prediction: {e}")

# Evaluate the model
try:
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
except Exception as e:
    print(f"Error during model evaluation: {e}")

# Function to optimize schedule based on model prediction
def optimize_schedule(employee_data):
    data_for_optimization = pd.get_dummies(employee_data, columns=['CurrentShift', 'PreferredShift', 'Performance'])
    data_for_optimization = data_for_optimization.reindex(columns=features.columns, fill_value=0)

    try:
        satisfaction_predictions = model.predict(data_for_optimization)
        employee_data['OptimizedSchedule'] = np.where(satisfaction_predictions == 1, employee_data['PreferredShift'], employee_data['CurrentShift'])
        return employee_data
    except Exception as e:
        print(f"Error during schedule optimization: {e}")
        return employee_data

# Example usage:
optimized_schedule = optimize_schedule(data.copy())
print("\nOptimized Schedule:")
print(optimized_schedule[['EmployeeID', 'CurrentShift', 'PreferredShift', 'OptimizedSchedule']])
```

### Key Points:
- **Data Preparation**: The script creates a synthetic dataset that includes features such as current shift, preferred shift, work hours, and performance. These are encoded for use in machine learning.
- **Model Training**: A simple `RandomForestClassifier` is used to demonstrate model training.
- **Error Handling**: Try-except blocks ensure that errors during training, prediction, or evaluation are caught and reported.
- **Schedule Optimization**: A function is provided to optimize schedules based on model predictions, with the aim of maximizing employee satisfaction (i.e., matching their preferred shift).

This program is simplified and for educational purposes. In a real-world application, additional data engineering, constraints handling, and model sophistication would be necessary.