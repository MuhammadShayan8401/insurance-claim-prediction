# 💰 Insurance Claim Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Objective

To predict medical insurance claim charges based on personal attributes such as age, BMI, smoking status, and region using Linear Regression and provide an interactive web-based dashboard for analysis and prediction.

---

## 🌐 Live Deployment

🚀 The application is deployed on Streamlit Cloud:

👉 **(Add your Streamlit app link here)**

---

## 📊 Dataset

Medical Cost Personal Dataset includes:

* Age
* Sex
* BMI
* Number of children
* Smoking status
* Region
* Insurance charges (target variable)

---

## ⚙️ Approach

* Data cleaning and preprocessing
* Encoding categorical variables (sex, smoker, region)
* Exploratory Data Analysis (EDA) using visualizations
* Feature correlation analysis
* Model training using Linear Regression
* Model evaluation using MAE and RMSE
* Interactive dashboard using Streamlit

---

## 📈 Features of Dashboard

* User input (sidebar form)
* Real-time insurance charge prediction
* BMI vs charges visualization
* Age vs charges visualization
* Smoking impact analysis
* Feature correlation heatmap
* Model evaluation metrics (MAE, RMSE)

---

## 🤖 Results

* Linear Regression model successfully predicts insurance charges
* Smoking status has the strongest impact on cost
* BMI shows moderate positive correlation with charges
* Age gradually increases insurance cost
* Model evaluated using:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)

---

## 🧠 Model Explanation

The Linear Regression model learns relationships between input features and insurance charges. Each feature contributes linearly:

* Smoking → Strong positive impact
* BMI → Moderate impact
* Age → Gradual increase
* Region → Minor variation

---

## 🛠️ Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

---

## 👨‍💻 Author

Muhammad Shayan Ahmed

---

## 📌 Key Insight

Smoking and BMI are the strongest factors affecting insurance charges in this dataset.
