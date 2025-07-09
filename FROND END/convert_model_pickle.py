import pickle

# Try loading using the legacy pickle5-compatible mode
input_file = "random_forest_model.pkl"
output_file = "random_forest_model_converted.pkl"

with open(input_file, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Re-save using current Python version
with open(output_file, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model successfully converted and saved as:", output_file)
