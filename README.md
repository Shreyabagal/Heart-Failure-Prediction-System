# Heart Failure Prediction System

## Description

The **Heart Failure Prediction System** is a machine learning-based web application built using Streamlit that predicts the risk of heart failure in a patient based on various health metrics and lifestyle data. The system utilizes a **Random Forest Classifier** model, which is trained on a dataset of medical and lifestyle features.

The application allows users to input personal details, medical examination data, and lifestyle information to predict the risk of heart failure and provides a detailed summary with a risk breakdown. The system helps in identifying individuals at risk of heart failure, thus enabling timely medical intervention.

## Features

- **User Input Form**: Allows users to input personal and lifestyle details such as age, gender, BMI, smoking status, physical activity, alcohol consumption, etc.
- **Heart Failure Risk Prediction**: Based on the input data, the model predicts the risk level (LOW, MODERATE, or HIGH) of heart failure.
- **Visualization**: The app provides visualizations of key health metrics and risk breakdown through bar charts and pie charts.
- **Patient Summary**: Displays a table summarizing the patient’s input data.
- **Risk Breakdown**: Explains key factors contributing to the risk of heart failure.

## Technologies Used

- **Streamlit**: For building the interactive web app interface.
- **Python**: For the machine learning model and data processing.
- **Scikit-learn**: For the Random Forest model and preprocessing.
- **Pandas**: For data manipulation and cleaning.
- **Matplotlib**: For data visualization.
- **LabelEncoder**: For encoding categorical variables.

## Dataset

The dataset used for training the model includes various health indicators such as age, blood pressure, cholesterol, BMI, and lifestyle factors like smoking, physical activity, and alcohol consumption. It is crucial for training the prediction model to assess heart disease risk based on historical health data.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Shreyabagal/Heart-Failure-Prediction-System.git
    cd heart-failure-prediction
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

4. **Enter the patient details** in the input form and click the "Predict" button to see the risk prediction and summary.

## Requirements

- Python 3.7+
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Shreyabagal/Heart-Failure-Prediction-System.git


**Step 1: Navigate to the Project Directory**
Open your terminal and navigate to the directory where your project is located:

cd heart-failure-prediction

**Step 2: Install the Dependencies**
Once inside the project directory, install all required dependencies from the requirements.txt file. This file contains a list of Python packages needed to run the project.

pip install -r requirements.txt

**Step 3: Run the Streamlit App**
To start the Streamlit app, run the following command in the terminal:

streamlit run app.py
This will open your web browser and launch the application. The app allows you to input data and receive predictions for heart failure risk.

**Model Explanation**
The heart failure prediction model is built using a Random Forest Classifier, a machine learning algorithm that helps in predicting the likelihood of heart failure based on input data. The model is trained on a dataset that includes various health metrics such as age, blood pressure, cholesterol, BMI, and lifestyle details like smoking and physical activity. After training, the model predicts the risk of heart failure based on the user input.

**Risk Levels**
The model provides three risk levels for heart failure:

LOW: The patient is at low risk of heart failure. Maintaining a healthy lifestyle is recommended.

MODERATE: The patient has a moderate risk. It is advised to have regular health screenings and consider preventive care.

HIGH: The patient is at high risk of heart failure and should immediately consult a healthcare provider for further evaluation and intervention.
