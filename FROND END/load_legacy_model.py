import pickle
import joblib
import io
import warnings

warnings.filterwarnings("ignore")

# Step 1: Read the old pickle data in binary mode
with open("random_forest_model.pkl", "rb") as f:
    raw_data = f.read()

# Step 2: Use pickle5 to load the data safely into memory
try:
    model = pickle.loads(raw_data)
    print("✅ Legacy model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

# Step 3: Save the model again using the current joblib version
try:
    joblib.dump(model, "random_forest_model_converted.pkl")
    print("✅ Model converted successfully to 'random_forest_model_converted.pkl'")
except Exception as e:
    print("❌ Failed to save converted model:", e)
