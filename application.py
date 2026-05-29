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

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Removes huge empty padding space at the top of the main container */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        /* Removes empty whitespace block sitting right above the main title */
        stHeader {
            height: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

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

#---------------------------------------------------------------
# Functions
#--------------------------------------------------------------

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "dashboard"

def go_to_simulator():
    st.session_state["current_page"] = "simulator"

def go_to_dashboard():
    st.session_state["current_page"] = "dashboard"

if "customer_data" not in st.session_state:
    st.session_state["customer_data"] = None

past_median_BALANCE = 97198.54
past_median_SALARY = 100193.915
train_data_count= 10000

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


@st.dialog("Upload Customer File")
def upload_file_dialog():
    st.write("Please select your bank customer CSV file.")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Run"):
            st.session_state["customer_data"] = pd.read_csv(uploaded_file)
            go_to_dashboard()
            st.rerun()
#---------------------------------------------------------
# General Dashboard
#---------------------------------------------------------
if st.session_state["current_page"] == "dashboard":
    st.title("General Dashboard")
    btn_left_col, space_col, btn_right_col = st.columns([1.5, 6, 2.5], vertical_alignment="top")
    with btn_left_col:
        st.button("What-IF simulator", on_click= go_to_simulator)
    with btn_right_col:
        if st.button("📥 Import Customer CSV Data"):
            upload_file_dialog()
    if st.session_state["customer_data"] is not None:
        customer = st.session_state["customer_data"]
        columns_need= ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
        columns_current_set= set(customer.columns)
        columns_need_set=  set(columns_need)
        has_all_columns = columns_need_set.issubset(columns_current_set)
        if has_all_columns:
            bank= customer[columns_need]
            new_median_balance = bank['Balance'].median()
            new_mdeian_salary = bank['EstimatedSalary'].median()
            n_new = len(bank)
            TRAINING_MEDIAN_BALANCE = (past_median_BALANCE * train_data_count + new_median_balance * n_new) / (train_data_count + n_new)
            TRAINING_MEDIAN_SALARY = (past_median_SALARY * train_data_count + new_mdeian_salary * n_new) / (train_data_count + n_new)

            bank= create_basic_features(bank)
            bank= create_intermediate_features(bank)
            bank= create_advanced_features(bank)
            bank= create_additional_features(bank)
            processed_bank = preprocessor.transform(bank)
            probabilities = classifier.predict_proba(processed_bank)
            churn_risk_scores = probabilities[:, 1]
            results_df = pd.DataFrame({ "CustomerId": customer["CustomerId"],"Churn Risk Score": churn_risk_scores})
            results_df["Churn Risk Score"] = results_df["Churn Risk Score"].map("{:.1%}".format)
            col_left, col_right = st.columns([4, 6])
            with col_left:
                if "visible_rows" not in st.session_state:
                    st.session_state["visible_rows"] = 10
                current_limit = st.session_state["visible_rows"]
                sorted_df = results_df.sort_values(by="Churn Risk Score", ascending=False)
                expanded_df = sorted_df.head(current_limit).copy()
                st.write(f"#### 🚨 Top {len(expanded_df)} High-Risk Customers (Most Likely to Churn)")
                st.dataframe(expanded_df, use_container_width=True, height= 150)
                total_available_rows = len(results_df)
                if current_limit < total_available_rows:
                    def load_more_customers():
                        st.session_state["visible_rows"] += 5
                    st.button("🔽 Click to see more", on_click=load_more_customers)
                else:
                    st.info("✨ Showing all available customer risk scores.")
                    
                st.markdown("<br>", unsafe_allow_html=True)
            
                st.write("#### 📊 Probability Distribution Visualization")
                fig_dist = px.histogram(
                results_df, 
                x="Churn Risk Score", 
                labels={"Churn Risk Score": "Predicted Churn Probability", "count": "Number of Customers"},
                color_discrete_sequence=["#4A90E2"]
                )
                fig_dist.update_layout(yaxis_title="Count of Customers", height= 250)
                st.plotly_chart(fig_dist, use_container_width=True)

                
            with col_right:
                importances = classifier.feature_importances_
                feature_names = preprocessor.get_feature_names_out()
                clean_feature_names = [name.split("__")[-1] for name in feature_names]
                df_importance = pd.DataFrame({
                "Feature": clean_feature_names,
                "Importance": importances
                }).sort_values(by="Importance", ascending=True)
                st.write("#### Key Drivers of Customer Churn")
                fig_importance = px.bar(
                df_importance,
                x="Importance",
                y="Feature",
                orientation="h",
                labels={"Importance": "Relative Importance Score", "Feature": "Customer Attribute"},
                color="Importance",
                color_continuous_scale="Blues")
                fig_importance.update_layout(yaxis={"categoryorder": "total ascending"},  height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
                st.info(
                    "💡 **Regulatory Insight:** This chart displays the global drivers of churn risk. "
                    "Higher scores indicate that the feature has a stronger impact on whether a customer stays or leaves."
                )
        else:
            missing_columns = columns_need_set - columns_current_set
            st.error(f"❌ Missing Columns! The uploaded file is missing: {list(missing_columns)}")
    else:
        st.write("Upload Customer data as .csv file. Ensure CustomerId ifeature is inside the CSV file.")

#----------------------------------------------------
# What-if Scenario simulator
#----------------------------------------------------
if st.session_state["current_page"] == "simulator":
    st.title("🧪 What-If Scenario Simulator")
    with st.sidebar:
        st.write("Adjust customer attributes to simulate churn risk behavior.")
        st.button("General Dashboard", on_click=go_to_dashboard)
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
    TRAINING_MEDIAN_BALANCE = (past_median_BALANCE * train_data_count + Balance * 1) / (train_data_count + 1)
    TRAINING_MEDIAN_SALARY = (past_median_SALARY * train_data_count + Salary * 1) / (train_data_count + 1)
    
    #------------------------
    # Feature Engineering
    #--------------------------
    
    
    bank= create_basic_features(bank)
    bank= create_intermediate_features(bank)
    bank= create_advanced_features(bank)
    bank= create_additional_features(bank)
    
    #----------------------------------------------------
    # Prediction
    #----------------------------------------------------
    bank_preprocessed= preprocessor.transform(bank)
    if st.button("Predict Churn Risk"):
        probability = classifier.predict_proba(bank_preprocessed)
        risk_percentage = probabilities[0][1]
        if risk_percentage < 30:
            color = "#2ecc71"  # Soft Green
            status_label = "🟢 Low Churn Risk"
        elif risk_percentage < 70:
            color = "#f39c12"  # Soft Orange
            status_label = "🟡 Medium Churn Risk"
        else:
            color = "#e74c3c"  # Soft Red
            status_label = "🔴 High Churn Risk!"

        st.write("### Prediction Results")
        st.markdown(f"""
            <div style="background-color: #1e222b; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                <p style="margin: 0; font-size: 14px; color: #a3a8b4; font-weight: bold; text-transform: uppercase;">{status_label}</p>
                <h1 style="margin: 5px 0 0 0; font-size: 48px; color: {color}; font-weight: bold;">{risk_percentage:.2f}%</h1>
            </div>
        """, unsafe_allow_html=True)  
        











