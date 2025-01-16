import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt



# Set up page configuration
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add custom CSS to increase the font size of the checkbox label and make it bold
st.markdown(
    """
    <style>
    /* Change the font size and make the checkbox label bold */
    .stCheckbox label {
        font-size: 24px; /* Adjust the font size */
        font-weight: bold; /* Make the text bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Heart Disease Risk - Prediction App")


# Main Page
st.header('What is Heart Attack?')
st.video('https://youtu.be/bw_Vv2WRG-A')

st.write("""
This app predicts whether a patient has heart disease based on user inputs and showcases EDA analysis.
""")
st.write("""This project is to create a model that able to make a prediction of heart attack possibilities in a patient. I have deployed an app using Streamlit platform. This project used Logistic Regression classification model of Machine Learning (ML) to predict the required results.""")


# Sidebar for user inputs
st.sidebar.header("Enter User Input Features Here : ")


def user_input_features():
    age = st.sidebar.number_input('Age:', min_value=0, max_value=120, value=30)
    sex = st.sidebar.selectbox('Sex:', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox(
        'Chest Pain Type:',
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x]
    )
    trtbps = st.sidebar.number_input('Resting Blood Pressure (mmHg):', min_value=80, max_value=200, value=120)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dL):', min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dL:', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
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
    exng = st.sidebar.selectbox('Exercise-Induced Angina:', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input('ST Depression:', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slp = st.sidebar.selectbox(
        'Slope of ST Segment:',
        options=[0, 1, 2],
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x]
    )
    caa = st.sidebar.selectbox('Number of Major Vessels:', options=[0, 1, 2, 3], format_func=lambda x: f"{x} Major Vessel(s)")
    thal = st.sidebar.selectbox(
        'Thalassemia:',
        options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x]
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



if st.checkbox("## **Show EDA Analysis 🔍**"):
    st.header("EDA Analysis")

    # Load dataset for EDA
    df = pd.read_csv('heart.csv')

    # Display dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Display dataset information
    st.subheader("Dataset Info")
    st.write(df.describe())
    st.write("### Conclusion:")
    st.write("The dataset contains numerical and categorical variables with no missing values, as seen in the summary statistics.")

    # Visualize missing values
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0])
    if missing.sum() == 0:
        st.write("No missing values detected in the dataset.")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    st.write("This heatmap shows the correlation between features.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.write("### Conclusion:")
    st.write(""" 
    - Strong correlation observed between certain features like `thalachh` (maximum heart rate achieved) and `output` (target variable).
    - Some features like `chol` (serum cholesterol) have weaker correlations with `output`.
    """)

    # Pairplot
    st.subheader("Pairplot")
    st.write("Pairplot of selected features.")
    fig = sns.pairplot(df, vars=["age", "chol", "thalachh", "trtbps", "oldpeak"], diag_kind="kde", hue="output")
    st.pyplot(fig)
    st.write("### Conclusion:")
    st.write(""" 
    - The pairplot reveals distinct separations between high and low heart disease risk based on certain features like `thalachh` and `oldpeak`.
    - Some overlaps exist, indicating potential challenges in classification for certain ranges of features.
    """)

    # Feature mapping dictionary for full descriptive names
    feature_full_names = {
        "age": "Age",
        "sex": "Sex",
        "cp": "Chest Pain Type",
        "trtbps": "Resting Blood Pressure (mmHg)",
        "chol": "Serum Cholesterol (mg/dL)",
        "fbs": "Fasting Blood Sugar > 120 mg/dL",
        "restecg": "Resting ECG Results",
        "thalachh": "Maximum Heart Rate Achieved (bpm)",
        "exng": "Exercise-Induced Angina",
        "oldpeak": "ST Depression",
        "slp": "Slope of ST Segment",
        "caa": "Number of Major Vessels",
        "thall": "Thalassemia",
        "output": "Heart Disease Risk"
    }

    # Feature Distributions
    st.subheader("Feature Distributions")
    feature = st.selectbox(
        "Select a feature for distribution plot:",
        df.columns.map(lambda col: feature_full_names[col])  # Map full names for display
    )

    # Map back the selected full name to the original column name
    original_feature = {v: k for k, v in feature_full_names.items()}[feature]

    # Plotting the distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[original_feature], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    st.pyplot(fig)

    # Dynamic conclusion for feature distribution
    st.write("### Conclusion:")
    if original_feature == "age":
        st.write("Age is normally distributed, with most patients being between 40 and 70 years old.")
    elif original_feature == "chol":
        st.write("Serum cholesterol levels are right-skewed, with most values between 200 and 300 mg/dL.")
    elif original_feature == "thalachh":
        st.write("Maximum heart rate achieved peaks around 150 bpm, indicating its importance for predicting heart disease.")
    elif original_feature == "output":
        st.write("The target variable is balanced, with a roughly equal number of cases with and without heart disease.")
    else:
        st.write(f"The feature `{feature}` has the above distribution, providing insights into its range and common values.")




# Model prediction logic
model_filename = 'Logistic_regression_model.joblib'

try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.warning("Model not found! Training a new model...")
    df = pd.read_csv('heart.csv')
    X = df.drop(columns=['output'])
    y = df['output']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)

if model:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    
    # Show success (green) for low risk and warning (red) for high risk
    if prediction[0] == 1:
        st.warning("High Risk", icon="⚠️")  # High risk shown in red
    else:
        st.success("Low Risk", icon="✅")  # Low risk shown in green

    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(
        prediction_proba,
        columns=["Not at Risk", "At Risk"]
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
