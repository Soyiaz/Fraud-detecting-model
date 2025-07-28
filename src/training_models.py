from load_datas import load_data
import joblib
import os

# Load the data
X_train_resampled, y_train, X_test, y_test = load_data()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import average_precision_score

# Train Logistic Regression
print("Training Logistic Regression model...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_resampled, y_train)
print("Logistic Regression model trained.")

# Train Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_resampled, y_train)
print("Random Forest model trained.")


os.makedirs("models", exist_ok=True)

joblib.dump(log_reg, "models/logistic_regression_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")

print("✅ Models saved successfully!")


