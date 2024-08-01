import pickle

try:
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file: {e}")

