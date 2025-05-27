from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

app = Flask(__name__)

try:
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: diabetes_model.pkl or scaler.pkl not found. Make sure they are in the same directory as app.py.")
    model = None
    scaler = None

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Basic authentication (replace with a proper system)
        username = request.form["username"]
        password = request.form["password"]
        if username == "user" and password == "password":  # Replace with your logic
            return redirect(url_for("dashboard"))
        else:
            return "Invalid username or password"  # Handle invalid login
    return render_template("home.html")

@app.route("/dashboard", methods=["GET", "POST"])  # Allow both GET and POST requests
def dashboard():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Basic registration (replace with a proper system)
        username = request.form["username"]
        password = request.form["password"]
        # Add your logic to store the username and password
        return redirect(url_for("home"))  # Redirect to login after registration
    return render_template("register.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST" and model and scaler:  # Check if model and scaler are loaded
        try:
            values = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"])
            ]
            input_data = np.array(values).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            prediction = "Diabetic" if pred == 1 else "Not Diabetic"
        except ValueError:
            prediction = "Invalid input. Please enter numeric values."
        except KeyError:
            prediction = "Missing input field. Please fill in all fields."

    return render_template("prediction_form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    
    import os
# ...
model_path = "diabetes_model.pkl"
scaler_path = "scaler.pkl"

print(f"Attempting to load model from: {os.path.abspath(model_path)}")
print(f"Attempting to load scaler from: {os.path.abspath(scaler_path)}")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: diabetes_model.pkl or scaler.pkl not found. Make sure they are in the same directory as app.py.")
    model = None
    scaler = None
except Exception as e:
    print(f"An error occurred while loading the model or scaler: {e}")
    model = None
    scaler = None