# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Define paths to your pickle files
# Ensure these files are in the same directory as app.py
GENDER_ENCODER_PATH = "gender-encoder.pkl"
DEGREE_ENCODER_PATH = "degree-encoder.pkl"
DEPRESSION_MODEL_PATH = "depression-lr-model.pkl"

# Load models and encoders
try:
    with open(GENDER_ENCODER_PATH, "rb") as f:
        gender_encoder = pickle.load(f)
    print(f"Successfully loaded gender encoder from {GENDER_ENCODER_PATH}")

    with open(DEGREE_ENCODER_PATH, "rb") as f:
        degree_encoder = pickle.load(f)
    print(f"Successfully loaded degree encoder from {DEGREE_ENCODER_PATH}")

    with open(DEPRESSION_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Successfully loaded depression model from {DEPRESSION_MODEL_PATH}")

except FileNotFoundError as e:
    print(f"Error: One or more pickle files not found. Please ensure they are in the same directory as app.py.")
    print(f"Missing file: {e.filename}")
    # Exit or handle the error gracefully, maybe display an error page
    exit() # Exiting for now, in a production app you'd render an error template
except Exception as e:
    print(f"An error occurred while loading pickle files: {e}")
    exit() # Exiting for now


# Prediction function
def predict_depression(gender, age, academic_pressure, sleep_hours, degree):
    try:
        # Encode categorical inputs using the loaded encoders
        # .transform returns an array, so we take the first element [0]
        gender_encoded = gender_encoder.transform([gender])[0]
        degree_encoded = degree_encoder.transform([degree])[0]

        # Ensure academic_pressure and sleep_hours are numbers
        academic_pressure = float(academic_pressure)
        sleep_hours = float(sleep_hours)

        # Prepare features list in the correct order as expected by your model
        # Based on your notebook, features used for training were likely:
        # Gender (encoded), Age, Academic Pressure, Sleep Duration (encoded), Degree (encoded)
        # Note: The notebook also processed 'Sleep Duration' into numeric.
        # This app's input 'sleep_hours' should map to that.
        # If your model needs other features that were in your X, you need to add them here.
        # For simplicity, I'm assuming 'Gender', 'Age', 'Academic Pressure', 'Sleep Duration', 'Degree'
        # are the primary features matching your input fields.
        
        # Check the exact features used for model training in your notebook
        # For example, if your X_train had columns like:
        # df = pd.read_csv("student_depression_dataset.csv")
        # # ... preprocessing for Sleep Duration, Suicidal thoughts, Family History, Financial Stress ...
        # # ... LabelEncoding for all object columns including Gender and Degree ...
        # # X = df.drop(columns=['Depression']) # This is crucial to know exact columns and their order

        # Let's assume the feature order after preprocessing in your notebook is:
        # [Gender, Age, Academic Pressure, Sleep Duration (numeric), Degree (encoded)]
        # You need to ensure 'sleep_hours' from user input is transformed like 'Sleep Duration' in your notebook.
        # Based on your notebook's Cell In[8] for 'sleep_duration' mapping:
        sleep_duration_map = {
            'Less than 5 hours': 4.0,
            '5-6 hours': 5.5,
            '7-8 hours': 7.5,
            'More than 8 hours': 9.0
        }
        # Assuming sleep_hours from input is a raw number (e.g., 5.5).
        # If your model expects the categorical mapping, you need to convert input sleep_hours to its mapped category first.
        # For now, let's assume `sleep_hours` from input is already a numeric representation.
        # If not, you need to map it based on ranges or categories from your original data.
        
        # IMPORTANT: The order of features MUST match the order used during model training.
        # You need to inspect the 'X' (features) dataframe in your notebook right before model.fit(X_train, y_train).
        # A common set of features after basic preprocessing based on your notebook snippets might be:
        # Gender (encoded), Age, Academic Pressure, Sleep Duration (numeric), Degree (encoded),
        # Have you ever had suicidal thoughts? (encoded), Work/Study Hours, Financial Stress (numeric),
        # Family History of Mental Illness (encoded), etc.

        # I will create a feature vector assuming a minimal set from your inputs for demonstration.
        # You MUST adjust this `features` array to match the exact order and number of features
        # your `depression-lr-model.pkl` expects.
        
        # Example assuming the model expects [gender_encoded, age, academic_pressure, sleep_hours, degree_encoded]
        features = [[gender_encoded, age, academic_pressure, sleep_hours, degree_encoded]]

        # If your model expects more features from the dataset, you need to provide default values or get them from the user
        # For example, if `Work/Study Hours`, `Financial Stress`, etc., are features:
        # features = [[
        #     gender_encoded,
        #     age,
        #     academic_pressure,
        #     sleep_hours,
        #     degree_encoded,
        #     # Add other features if your model was trained on them:
        #     # suicidal_thoughts_encoded (e.g., 0 for No),
        #     # work_study_hours_default (e.g., 8.0),
        #     # financial_stress_numeric (e.g., 0.0),
        #     # family_history_encoded (e.g., 0 for No)
        # ]]


        prediction = model.predict(np.array(features))[0] # Use np.array for consistent input

        # Interpret the prediction
        return "Person is likely depressed." if prediction == 1 else "Person is not depressed."

    except ValueError as e:
        return f"Input Error: {e}. Please check your input values."
    except Exception as e:
        return f"An unexpected error occurred during prediction: {e}"

# Home route with form
@app.route("/", methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_values = {
        'gender': 'Male',
        'age': 25,
        'academic_pressure': 3,
        'sleep_hours': 7.0,
        'degree': 'Undergraduate'
    }

    if request.method == 'POST':
        try:
            gender = request.form.get('gender')
            age = int(request.form.get('age'))
            academic_pressure = int(request.form.get('academic_pressure'))
            sleep_hours = float(request.form.get('sleep_hours'))
            degree = request.form.get('degree')

            # Update input_values to display last entered data
            input_values = {
                'gender': gender,
                'age': age,
                'academic_pressure': academic_pressure,
                'sleep_hours': sleep_hours,
                'degree': degree
            }

            # Validate input against encoder classes if possible
            if gender not in gender_encoder.classes_:
                prediction_result = f"Error: Invalid gender! Choose from: {list(gender_encoder.classes_)}"
            elif degree not in degree_encoder.classes_:
                prediction_result = f"Error: Invalid degree! Choose from: {list(degree_encoder.classes_)}"
            else:
                prediction_result = predict_depression(gender, age, academic_pressure, sleep_hours, degree)

        except ValueError:
            prediction_result = "Error: Please ensure age, academic pressure, and sleep hours are valid numbers."
        except Exception as e:
            prediction_result = f"An error occurred: {e}"

    # Provide common choices for dropdowns/inputs
    # These should match the classes your encoders were trained on!
    # You might need to retrieve these directly from the loaded encoders or your training script.
    available_genders = list(gender_encoder.classes_) if 'gender_encoder' in globals() else ['Male', 'Female', 'Other']
    available_degrees = list(degree_encoder.classes_) if 'degree_encoder' in globals() else ['Undergraduate', 'Postgraduate', 'B.Sc.', 'M.Sc.', 'PhD', 'B.Tech', 'M.Tech', 'B.Pharm', 'MD', 'BCA', 'BA', 'Class 12'] # Add more based on your dataset

    return render_template(
        'index.html',
        prediction=prediction_result,
        input_values=input_values,
        available_genders=available_genders,
        available_degrees=available_degrees
    )

# Other placeholder routes (you can add content to their respective HTML files)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

if __name__ == '__main__':
    # Flask runs on 127.0.0.1:5000 by default.
    # debug=True allows for automatic reloading on code changes and provides more detailed error messages.
    app.run(debug=True)

