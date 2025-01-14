import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

st.title("Heart Disease Prediction App")

st.write("""
This app predicts whether a patient has heart disease based on user inputs.
Data Source: [Kaggle Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
""")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.number_input('Age:', min_value=0, max_value=120, value=30)
    sex = st.sidebar.selectbox('Sex (0: Female, 1: Male):', options=[0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (0-3):', options=[0, 1, 2, 3])
    trtbps = st.sidebar.number_input('Resting Blood Pressure (mmHg):', min_value=80, max_value=200, value=120)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dL):', min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dL (1: Yes, 0: No):', options=[0, 1])
    restecg = st.sidebar.selectbox('Resting ECG Results (0-2):', options=[0, 1, 2])
    thalachh = st.sidebar.number_input('Max Heart Rate Achieved:', min_value=50, max_value=250, value=150)
    exng = st.sidebar.selectbox('Exercise-Induced Angina (1: Yes, 0: No):', options=[0, 1])
    oldpeak = st.sidebar.number_input('ST Depression:', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slp = st.sidebar.selectbox('Slope of ST Segment (0-2):', options=[0, 1, 2])
    caa = st.sidebar.selectbox('Number of Major Vessels (0-3):', options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversible Defect):', options=[0, 1, 2])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thal
    }
    return pd.DataFrame(data, index=[0])

# Gather user inputs
input_df = user_input_features()

# Display user inputs
st.write("### User Input Features")
st.write(input_df)

# Check if the model exists, and if not, train and save it
model_filename = 'Logistic_regression_model.joblib'

try:
    # Try to load the pre-trained model
    model = joblib.load(model_filename)
    st.success("Model loaded successfully.")
except FileNotFoundError:
    st.warning("Model not found! Training a new model...")

    # Load dataset for training
    df = pd.read_csv('heart.csv')
    X = df.drop(columns=['output'])
    y = df['output']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Save the model for future use
    joblib.dump(model, model_filename)
    st.success("New model trained and saved.")

    # Display accuracy for the newly trained model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Apply the model to make predictions
if model:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    st.write("You Have High Chance Of Getting Heart Attack" if prediction[0] == 1 else "You Have Low Chance Of Getting Heart Attack")

    st.subheader("Prediction Probability")
    st.write(prediction_proba)
