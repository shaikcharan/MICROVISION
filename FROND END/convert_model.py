import joblib
import sklearn

print("Current scikit-learn version:", sklearn.__version__)

# Load the old model
model = joblib.load("random_forest_model.pkl")

# Save it again using the current version (1.7.2)
joblib.dump(model, "random_forest_model_converted.pkl")

print("✅ Model converted successfully and saved as random_forest_model_converted.pkl")
