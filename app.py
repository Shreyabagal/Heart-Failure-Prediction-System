import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Heart Failure Prediction System", layout="centered")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("Heart Failure Prediction System.csv")
    df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "") for col in df.columns]
    return df

df = load_data()

# Preprocess
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(['Patient_ID', 'Heart_Disease_0__No_1__Yes'], axis=1)
y = df['Heart_Disease_0__No_1__Yes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# App Styling
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main-title {
        font-size: 48px;
        font-weight: 900;
        color: red;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: white;
        background-color: #222;
        padding: 10px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .highlight {
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        margin-top: 15px;
    }
    .low { background-color: #b7f0ad; color: #1b4332; }
    .moderate { background-color: #ffe066; color: #664d03; }
    .high { background-color: #ff6b6b; color: #6a040f; }
    .stTextInput > div > input {
        border: 1px solid #aaa;
        border-radius: 6px;
        padding: 8px;
    }
    .stSelectbox > div {
        border-radius: 6px;
        padding: 6px;
    }
    .custom-predict-btn button {
        background: linear-gradient(to right, #ff4d6d, #f9dcc4);
        border: none;
        color: #1d3557;
        padding: 14px 30px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease-in-out;
    }
    .custom-predict-btn button:hover {
        transform: scale(1.05);
        background: linear-gradient(to right, #f9dcc4, #ff4d6d);
        color: #e63946;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🩺 Heart Failure Prediction System</div>', unsafe_allow_html=True)

# Added line
st.markdown("### 📄 Enter Patient Details:")

# Input Form
st.markdown('<div class="section-title">🧍 Personal & Lifestyle Details</div>', unsafe_allow_html=True)

with st.form("input_form"):
    age = st.number_input("Age (Years)", min_value=20, max_value=80, value=40, help="Age of the patient in years")
    gender = st.radio("Gender", ["Female (F)", "Male (M)"], help="Select patient's gender")
    bmi = st.slider("BMI", min_value=15.0, max_value=40.0, value=25.0, help="Body Mass Index calculated from height and weight")
    smoking = st.radio("Smoking Status", ["Yes", "No"], help="Whether the patient smokes or not")
    activity = st.selectbox("Physical Activity Level", label_encoders["Physical_Activity_Level"].classes_, help="Patient's routine physical activity level")
    alcohol = st.selectbox("Alcohol Consumption", label_encoders["Alcohol_Consumption"].classes_, help="Frequency of alcohol consumption")
    fitness = st.slider("Fitness Score", min_value=0, max_value=100, value=50, help="Overall fitness level on a scale from 0 to 100")
    diabetes = st.selectbox("Diabetes Risk", label_encoders["Diabetes_Risk_Level"].classes_, help="Risk of diabetes based on health indicators")
    chest_pain = st.selectbox("Chest Pain Type", [
        "Typical Angina (TA)",
        "Atypical Angina (ATA)",
        "Non-Anginal Pain (NAP)",
        "Asymptomatic (ASY)"
    ], help="TA: common chest pain during effort\nATA: unusual chest pain\nNAP: not related to heart\nASY: no symptoms")

    st.markdown('<div class="section-title">🧬 Medical Examination Details</div>', unsafe_allow_html=True)

    bp = st.number_input("Blood Pressure (mmHg)", min_value=90, max_value=200, value=120, help="Resting blood pressure in mmHg")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, help="Cholesterol level in blood (mg/dL)")
    blood_sugar = st.selectbox("Fasting Blood Sugar", ["Normal", "High"], help="Fasting blood sugar levels: Normal or High")
    ecg = st.selectbox("Resting ECG Result", [
        "Normal",
        "ST-T Wave Abnormality (ST)",
        "Left Ventricular Hypertrophy (LVH)"
    ], help="Normal: normal result\nST: abnormal wave\nLVH: thickened left ventricle")
    max_hr = st.slider("Maximum Heart Rate", min_value=60, max_value=220, value=150, help="Maximum heart rate during exercise")
    angina = st.radio("Exercise-Induced Angina", ["Yes (Y)", "No (N)"], help="Chest pain during exercise? Yes or No")
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=0.0, help="ST segment depression during exercise (in mm)")
    st_slope = st.selectbox("ST Segment Slope", [
        "Down",
        "Flat",
        "Up"
    ], help="Down: downward slope\nFlat: no slope\nUp: upward slope")

    st.markdown('<div class="custom-predict-btn">', unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict")
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    with st.spinner("🔎 Analyzing patient data..."):
        gender_val = 0 if "F" in gender else 1
        smoking_val = label_encoders["Smoking_Status_Yes/No"].transform([smoking])[0]
        activity_val = label_encoders["Physical_Activity_Level"].transform([activity])[0]
        alcohol_val = label_encoders["Alcohol_Consumption"].transform([alcohol])[0]
        diabetes_val = label_encoders["Diabetes_Risk_Level"].transform([diabetes])[0]
        chest_pain_val = ["TA", "ATA", "NAP", "ASY"].index([x for x in ["TA", "ATA", "NAP", "ASY"] if x in chest_pain][0])
        ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
        ecg_val = ecg_map[[key for key in ecg_map if key in ecg][0]]
        angina_val = 0 if "N" in angina else 1
        blood_sugar_val = 1 if blood_sugar == "High" else 0
        st_slope_val = label_encoders["ST_Segment_Slope"].transform([st_slope])[0]

        input_data = [
            age, gender_val, bmi, smoking_val, activity_val,
            fitness, diabetes_val, chest_pain_val, bp, cholesterol,
            blood_sugar_val, ecg_val, max_hr, angina_val,
            st_depression, st_slope_val,
            label_encoders["Lifestyle"].transform(["Moderate exercise, balanced diet, occasional smoking"])[0],
            alcohol_val
        ]

        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        raw_prob = model.predict_proba(scaled_input)[0][1]
        prob = round(float(raw_prob) * 100, 2)

        risk_level = "LOW" if prediction == 0 else ("HIGH" if prob > 85 else "MODERATE")
        css_class = "low" if risk_level == "LOW" else ("high" if risk_level == "HIGH" else "moderate")

    st.markdown('<div class="section-title">🧾 Patient Summary</div>', unsafe_allow_html=True)

    summary_labels = [
        "Age", "Gender", "BMI", "Smoking Status", "Physical Activity",
        "Alcohol Consumption", "Fitness Score", "Diabetes Risk", "Chest Pain Type",
        "Blood Pressure", "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
        "Max Heart Rate", "Exercise-Induced Angina", "ST Depression", "ST Segment Slope"
    ]
    summary_values = [
        f"{age} years", gender, bmi, smoking, activity, alcohol, fitness,
        diabetes, chest_pain, f"{bp} mmHg", f"{cholesterol} mg/dL", blood_sugar,
        ecg, max_hr, angina, st_depression, st_slope
    ]

    summary_table = pd.DataFrame({
        "Parameter": summary_labels,
        "Value": summary_values
    })
    st.table(summary_table)

    st.markdown('<div class="section-title">📋 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='highlight {css_class}'>Risk Type: {risk_level}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='highlight {css_class}'>Risk Percentage: {prob}%</div>", unsafe_allow_html=True)

    advice_text = {
        "LOW": "🟢 Maintain a healthy lifestyle. Keep exercising regularly and stay hydrated.",
        "MODERATE": "🟠 Consider regular health screenings and consult a cardiologist for early preventive care.",
        "HIGH": "🔴 Immediate consultation with a healthcare provider is recommended. Adopt a heart-healthy lifestyle."
    }

    st.markdown(f"<div class='highlight {css_class}'>{advice_text[risk_level]}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔍 Risk Breakdown: Key Health Contributors</div>', unsafe_allow_html=True)

    explanation = []
    if age >= 55:
        explanation.append("🔺 Age over 55 increases heart risk.")
    if smoking == "Yes":
        explanation.append("🔺 Smoking is a major contributor to heart failure.")
    if cholesterol >= 240:
        explanation.append("🔺 High cholesterol can clog arteries.")
    if blood_sugar == "High":
        explanation.append("🔺 High blood sugar indicates possible diabetes, which stresses the heart.")
    if bp >= 140:
        explanation.append("🔺 Elevated blood pressure strains the heart.")
    if st_depression > 1.0:
        explanation.append("🔺 Significant ST depression can be a sign of ischemia.")
    if chest_pain == "Asymptomatic (ASY)":
        explanation.append("🔺 Asymptomatic chest pain can hide silent heart issues.")
    if fitness < 40:
        explanation.append("🔺 Low fitness levels are linked to poor cardiovascular health.")
    if not explanation:
        explanation.append("✅ All major risk factors are within a healthy range.")

    for e in explanation:
        st.markdown(f"<div class='highlight {css_class}'>{e}</div>", unsafe_allow_html=True)

        # 📊 Refined Professional Visualization of Patient Summary
    st.markdown('<div class="section-title">📈 Patient Health Data Overview</div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt

    # Metrics for bar chart
    bar_metrics = {
        "Age (yrs)": age,
        "BMI": bmi,
        "Blood Pressure": bp,
        "Cholesterol": cholesterol,
        "Max Heart Rate": max_hr,
        "Fitness Score": fitness,
        "ST Depression": st_depression
    }

    # Pie chart categories
    pie_labels = [
        f"Gender: {gender}",
        f"Smoking: {smoking}",
        f"Blood Sugar: {blood_sugar}",
        f"Angina: {angina}",
        f"Chest Pain: {chest_pain}",
        f"Diabetes Risk: {diabetes}",
        f"Activity: {activity}"
    ]
    pie_sizes = [1] * len(pie_labels)  # Equal size for display purpose

    # Create side-by-side layout
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#f5f7fa')  # Match Streamlit background
    fig.subplots_adjust(wspace=0.4)

    # --- Horizontal Bar Chart ---
    bar_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#17becf']
    bars = axes[0].barh(list(bar_metrics.keys()), list(bar_metrics.values()), color=bar_colors)
    axes[0].invert_yaxis()
    axes[0].set_title("Quantitative Health Metrics", fontsize=14, fontweight='bold', color='#1d3557')
    axes[0].set_xlabel("Value", fontsize=12)
    axes[0].tick_params(axis='y', labelsize=10)

    # Add value annotations on bars
  
    
  
  
  
    for bar in bars:
        width = bar.get_width()
        axes[0].text(width + 1, bar.get_y() + bar.get_height()/2,
                     f'{width:.1f}', va='center', fontsize=10, color='black')

    # --- Pie Chart for Categories ---
    pie_colors = plt.cm.Paired(np.linspace(0, 1, len(pie_labels)))
    wedges, texts = axes[1].pie(pie_sizes, labels=pie_labels, startangle=90, colors=pie_colors,
                                textprops={'fontsize': 10}, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
    axes[1].set_title("Categorical Health Profile", fontsize=14, fontweight='bold', color='#1d3557')

    # Render charts
    st.pyplot(fig)
