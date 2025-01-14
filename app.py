import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from PIL import Image

# Set up page configuration
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Heart Disease Risk - Prediction App")

# Open the image using PIL
img = Image.open(r"logo.png")

# Resize the image using PIL
img = img.resize((600, 600))  # Set the desired width and height here

# Display the resized image in Streamlit
st.image(img)


st.write("""
This app predicts whether a patient has heart disease based on user inputs.
""")

st.write("""This project is to create a model that able to make a prediction of heart attack possibilities in a patient. I have deployed an app using Streamlit platform. This project used Logistic Regression classification model of Machine Learning (ML) to predict the required results.""")

# Sidebar for user inputs
st.sidebar.header("Enter User Input Features Here : ")

def user_input_features():
    import streamlit as st

    age = st.sidebar.number_input('Age:', min_value=0, max_value=120, value=30)
    
    sex = st.sidebar.selectbox(
        'Sex:',
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    
    cp = st.sidebar.selectbox(
        'Chest Pain Type:',
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        }[x]
    )
    
    trtbps = st.sidebar.number_input('Resting Blood Pressure (mmHg):', min_value=80, max_value=200, value=120)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dL):', min_value=100, max_value=600, value=200)
    
    fbs = st.sidebar.selectbox(
        'Fasting Blood Sugar > 120 mg/dL:',
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    restecg = st.sidebar.selectbox(
        'Resting ECG Results:',
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "Having ST-T Wave Abnormality",
            2: "Showing Probable/Definite Left Ventricular Hypertrophy"
        }[x]
    )
    
    thalachh = st.sidebar.number_input('Max Heart Rate Achieved:', min_value=50, max_value=250, value=150)
    
    exng = st.sidebar.selectbox(
        'Exercise-Induced Angina:',
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    oldpeak = st.sidebar.number_input('ST Depression:', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    slp = st.sidebar.selectbox(
        'Slope of ST Segment:',
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )
    
    caa = st.sidebar.selectbox(
        'Number of Major Vessels:',
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} Major Vessel(s)"
    )
    
    thal = st.sidebar.selectbox(
        'Thalassemia:',
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect"
        }[x]
    )

    
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


# Main Page
st.header('What is Heart Attack?')
st.video('https://youtu.be/bw_Vv2WRG-A')



# Check if the model exists, and if not, train and save it
model_filename = 'Logistic_regression_model.joblib'

try:
    # Try to load the pre-trained model
    model = joblib.load(model_filename)
    st.write("Check Your Result Below : ")
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

if model:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    st.success(
        "You Have High Chance Of Getting Heart Attack"
        if prediction[0] == 1
        else "You Have Low Chance Of Getting Heart Attack"
    )

    st.subheader("Prediction Probability")
    # Create a labeled DataFrame for better readability
    prob_df = pd.DataFrame(
        prediction_proba,
        columns=["Prob of Not Getting Heart Attack", "Prob of Getting Heart Attack"]
    )
    st.write(prob_df)


# Footer
st.markdown(
    """
    <br><br>
    <footer style="text-align: center; font-size: 14px; color: gray;">
        Created with ❤️ by Hymavathi and The Team | <a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset" target="_blank">Dataset Source</a>
    </footer>
    """, unsafe_allow_html=True
)
