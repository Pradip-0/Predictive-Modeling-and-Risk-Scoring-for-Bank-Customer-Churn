import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import joblib
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
import json
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay
from itertools import combinations
import traceback
import plotly.express as px
import streamlit as st

#------------------------------
# Data and model loading
#------------------------------

try:
    classifier = joblib.load("Models/classifier.joblib")
except ModuleNotFoundError as e:
    st.error(f"Missing Library Detected: {e}")
    st.stop()
except Exception as e:
    st.error("Different Error Occurred:")
    st.code(traceback.format_exc())

try:
    preprocessor= joblib.load("Models/preprocessor.joblib")
except ModuleNotFoundError as e:
    st.error(f"Missing Library Detected: {e}")
except Exception as e:
    st.error("Different Error Occurred:")
    st.code(traceback.format_exc())
    
#-----------------------------------------
# Collecting Inputs
#-------------------------------------------
CreditScore = st.number_input("Enter your Credit Score")
Geography = st.selectbox("Enter location",options= ["France","Germany","Spain"])
Gender = st.radio("Gender",["Male","Female"])
Age = st.number_input("Enter Age",0,150, value= 30)
Tenure = st.slider("Enter Tenure", 0, 30, value= 1)
Balance_text = st.text_input("Enter Account Balance", value= 0.0)
try:
    Balance = float(Balance_text)
except ValueError:
    Balance = 0.0
  
NumOfProducts = st.slider("Number of products using", 0, 5)
HasCard_text= st.radio("Does the customer have a credit card?", ["Yes", "No"])
if HasCard_text == "Yes":
    HasCard = 1
else:
    HasCard = 0

IsActiveMember_text= st.radio("Does the customer use the services frequently?", ["Yes", "No"])
if IsActiveMember_text == "Yes":
  IsActiveMember = 1
else:
  IsActiveMember = 0
Salary= st.text_input("Enter Customer Salary", value= 10000)
try:
    Salary = float(Balance_text)
except ValueError:
    Salary = 0.0

input_data = {
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [Gender],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCard],  # Using standard feature name
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary":[Salary]
    }

bank= pd.DataFrame(input_data)

#------------------------
# Feature Engineering
#--------------------------
TRAINING_MEDIAN_BALANCE = 97198.54
TRAINING_MEDIAN_SALARY = 100193.915

def create_basic_features(df):
  df['AgeGroup'] = pd.cut(df['Age'], bins = [0, 19, 35, 60, 120], labels = ['Teenager', 'Young', 'Mid-age', 'Old'])
  df['IsZeroBalance'] = df['Balance'].apply(lambda x: 1 if x > 0 else 0)
  df['CreditScoreRating'] = pd.cut(df['CreditScore'], bins=[300, 579, 669, 739, 799, 850], labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
  df['HasMultipleProduct'] = df['NumOfProducts'].apply(lambda x: 1 if x > 1 else 0)
  return df

def create_intermediate_features(df):
  df['BalanceSalaryRatio']= df['Balance']/df['EstimatedSalary']
  df['TenureAgeRatio']= df['Tenure']/df['Age']
  df['CreditScoreAgeRatio']= df['CreditScore']/df['Age']
  df['EstimatedMonthlySalary']= df['EstimatedSalary']/12
  return df

def create_advanced_features(df):
    df["AgeBalanceInteraction"] = df["Age"] * df["Balance"]
    df["ProductDensity"] = df.apply(
        lambda row: (
            row["NumOfProducts"] / row["Tenure"] if row["Tenure"] != 0 else 0
        ),
        axis=1,
    )
    df["LoyalityScore"] = df["Tenure"] * df["IsActiveMember"]
    df["WealthSignifier"] = (
        (df["Balance"] > TRAINING_MEDIAN_BALANCE)
        & (df["EstimatedSalary"] > TRAINING_MEDIAN_SALARY)
    ).astype(int)
    return df

def create_additional_features(df):
    df["TenureRiskGroup"] = pd.cut(df["Tenure"],bins=[-1, 2, 9, np.inf],
        labels=["Early_HighRisk", "Mid_Stable", "Late_HighRisk"],
    )
    df["MidTierDanger"] = df["Balance"].between(100000, 150000).astype(int)
    df["GermanPassiveWealth"] = ((df["Geography"] == "Germany") & (df["IsActiveMember"] == 0) & (df["Balance"] > 0)).astype(int)
    df["HighRiskGermanCohort"] = ((df["Geography"] == "Germany") & (df["Age"] >= 45) & (df["IsActiveMember"] == 0) & (df["NumOfProducts"].isin([1, 3, 4]))).astype(int)
    df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"]
    df["CardButInactive"] = ((df["HasCrCard"] == 1) & (df["IsActiveMember"] == 0)).astype(int)
    df["CardAndActive"] = ((df["HasCrCard"] == 1) & (df["IsActiveMember"] == 1)).astype(int)
    df["TeenZeroBalance"] = ((df["Age"] <= 19) & (df["Balance"] == 0)).astype(int)
    df["ActiveZeroBalance"] = ((df["Balance"] == 0) & (df["IsActiveMember"] == 1)).astype(int)
    df["Female_Germany"] = ((df["Gender"] == "Female") & (df["Geography"] == "Germany")).astype(int)
    return df

bank= create_basic_features(bank)
bank= create_intermediate_features(bank)
bank= create_advanced_features(bank)
bank= create_additional_features(bank)

#----------------------------------------------------
# Prediction
#----------------------------------------------------
bank_preprocessed= preprocessor.transform(bank)
if st.button("Predict Churn Risk"):
    probabilities = classifier.predict_proba(bank_preprocessed)
    churn_probability = probabilities[0][1]
    st.write("### Prediction Results")
    st.metric(label="Risk Probability", value=f"{churn_probability:.2%}")
    st.progress(float(churn_probability))
    class_names = classifier.classes_
    df_prediction = pd.DataFrame({
    "Outcome": ["No Churn", "Churn Risk"],
    "Probability": [probabilities[0][0], probabilities[0][1]]
    })
    fig = px.bar(df_prediction, x="Outcome", y="Probability", text_auto=".1%", range_y=[0, 1])
    st.write(f"### Probability Distribution for Customer #{row_index}")
    st.plotly_chart(fig)













