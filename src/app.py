import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import train_model

st.set_page_config(
    page_title="Insurance AI Dashboard",
    page_icon="💰",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    💰 Insurance Claim Prediction Dashboard
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Linear Regression Model • Medical Cost Estimation
    </p>
    """,
    unsafe_allow_html=True
)

model, X_test, y_test, df, feature_cols = train_model()

st.sidebar.header("🧾 Customer Information")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

sex = 0 if sex == "male" else 1
smoker = 1 if smoker == "yes" else 0

region_map = {
    "southwest": [0, 0, 0],
    "southeast": [1, 0, 0],
    "northwest": [0, 1, 0],
    "northeast": [0, 0, 1]
}

input_data = np.array([[age, sex, bmi, children, smoker] + region_map[region]])

# FIX feature names warning
input_df = pd.DataFrame(input_data, columns=feature_cols)

st.markdown("## 📊 Prediction Result")

if st.button("🚀 Predict Insurance Cost"):
    prediction = model.predict(input_df)[0]

    st.success("Prediction Completed!")

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:#1f77b4;
            color:white;
            text-align:center;
            font-size:24px;">
            💰 Estimated Charges: ${prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.markdown("## 📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background: linear-gradient(135deg, #1f4037, #99f2c8);
            color:white;
            text-align:center;
            box-shadow: 2px 4px 10px rgba(0,0,0,0.2);
        ">
            <h3 style="margin:0;">📉 MAE</h3>
            <h2 style="margin:10px 0;">{mae:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            color:white;
            text-align:center;
            box-shadow: 2px 4px 10px rgba(0,0,0,0.2);
        ">
            <h3 style="margin:0;">📉 RMSE</h3>
            <h2 style="margin:10px 0;">{rmse:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## 📊 Data Analysis Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "BMI Impact",
    "Age Impact",
    "Smoking Impact",
    "Correlation"
])

# ---- BMI ----
with tab1:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", ax=ax)
    ax.set_title("BMI vs Charges")
    st.pyplot(fig)

# ---- AGE ----
with tab2:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", ax=ax)
    ax.set_title("Age vs Charges")
    st.pyplot(fig)

# ---- SMOKING ----
with tab3:
    fig, ax = plt.subplots()
    sns.boxplot(x="smoker", y="charges", data=df, ax=ax)
    ax.set_title("Smoking Impact")
    st.pyplot(fig)

# ---- CORRELATION (FIXED) ----
with tab4:
    numeric_df = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)


st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Made for Machine Learning Task 4 • Linear Regression Model</p>",
    unsafe_allow_html=True
)