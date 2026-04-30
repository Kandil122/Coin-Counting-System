import os
import joblib

# Get the directory where model.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'coin_svm_model.pkl')

model = joblib.load(model_path)
