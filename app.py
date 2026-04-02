import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------- Load model and columns ----------
model = joblib.load("customer_response_model.pkl")
columns = joblib.load("columns.pkl")  # List of all columns used in training

# ---------- Page Config ----------
st.set_page_config(
    page_title="Campaign Response Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

# ---------- Sidebar Inputs ----------
st.sidebar.header("Customer Inputs")
year_birth = st.sidebar.number_input("Year of Birth", min_value=1900, max_value=2025, value=1985)
income = st.sidebar.number_input("Income", min_value=0, value=50000)
recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=0, max_value=100, value=10)
mnt_wines = st.sidebar.number_input("Amount spent on Wines", min_value=0, value=500)
mnt_fruits = st.sidebar.number_input("Amount spent on Fruits", min_value=0, value=200)
mnt_meat = st.sidebar.number_input("Amount spent on Meat Products", min_value=0, value=400)
mnt_fish = st.sidebar.number_input("Amount spent on Fish Products", min_value=0, value=200)
mnt_sweets = st.sidebar.number_input("Amount spent on Sweets", min_value=0, value=150)
mnt_gold = st.sidebar.number_input("Amount spent on Gold Products", min_value=0, value=100)
num_deals = st.sidebar.number_input("Number of Deals Purchases", min_value=0, value=2)
num_web = st.sidebar.number_input("Number of Web Purchases", min_value=0, value=5)
num_catalog = st.sidebar.number_input("Number of Catalog Purchases", min_value=0, value=3)
num_store = st.sidebar.number_input("Number of Store Purchases", min_value=0, value=2)
num_web_visits = st.sidebar.number_input("Number of Web Visits per Month", min_value=0, value=5)
kid_home = st.sidebar.number_input("Number of Kids at Home", min_value=0, value=0)
teen_home = st.sidebar.number_input("Number of Teens at Home", min_value=0, value=0)
education = st.sidebar.selectbox("Education", ["Basic", "Graduation", "Master", "PhD"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Together", "Widow", "Alone", "YOLO"])

# ---------- Prepare Input Data ----------
input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Numeric features
input_data['Year_Birth'] = year_birth
input_data['Income'] = income
input_data['Recency'] = recency
input_data['MntWines'] = mnt_wines
input_data['MntFruits'] = mnt_fruits
input_data['MntMeatProducts'] = mnt_meat
input_data['MntFishProducts'] = mnt_fish
input_data['MntSweetProducts'] = mnt_sweets
input_data['MntGoldProds'] = mnt_gold
input_data['NumDealsPurchases'] = num_deals
input_data['NumWebPurchases'] = num_web
input_data['NumCatalogPurchases'] = num_catalog
input_data['NumStorePurchases'] = num_store
input_data['NumWebVisitsMonth'] = num_web_visits
input_data['Kidhome'] = kid_home
input_data['Teenhome'] = teen_home
input_data['Total_Spending'] = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweets + mnt_gold

# Categorical features
edu_col = f"Education_{education}"
mar_col = f"Marital_Status_{marital_status}"
if edu_col in input_data.columns:
    input_data[edu_col] = 1
if mar_col in input_data.columns:
    input_data[mar_col] = 1

# ---------- Prediction ----------
if st.button("Predict Response"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # ---------- Dashboard Layout ----------
    st.title("📊 Customer Response Dashboard")

    # Top metrics
    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.metric("Response Likely ✅", "Yes")
        col2.metric("Probability", f"{prob*100:.1f}%")
    else:
        col1.metric("Response Likely ❌", "No")
        col2.metric("Probability", f"{prob*100:.1f}%")

    # Spending Breakdown Pie Chart
    spending = {
        "Wines": mnt_wines,
        "Fruits": mnt_fruits,
        "Meat": mnt_meat,
        "Fish": mnt_fish,
        "Sweets": mnt_sweets,
        "Gold": mnt_gold
    }
    df_spending = pd.DataFrame(list(spending.items()), columns=["Category", "Amount"])
    fig1 = px.pie(df_spending, names='Category', values='Amount', title="Customer Spending Breakdown")
    st.plotly_chart(fig1, use_container_width=True)

    # Purchase Behavior Bar Chart
    purchases = {
        "Deals": num_deals,
        "Web Purchases": num_web,
        "Catalog Purchases": num_catalog,
        "Store Purchases": num_store,
        "Web Visits/Month": num_web_visits
    }
    df_purchases = pd.DataFrame(list(purchases.items()), columns=["Purchase Type", "Count"])
    fig2 = px.bar(df_purchases, x='Purchase Type', y='Count', color='Count', title="Customer Purchase Behavior")
    st.plotly_chart(fig2, use_container_width=True)

    # Age and Household Info
    age = 2026 - year_birth
    col3, col4 = st.columns(2)
    col3.metric("Customer Age", f"{age} years")
    col4.metric("Household", f"{kid_home} kids, {teen_home} teens")

    st.info("💡 Insights: Higher spending on Wines & Meat indicates premium preference. Frequent web visits may increase campaign response probability.")