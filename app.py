import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------- Page Config ----------
st.set_page_config(
    page_title="Campaign Response Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

# ---------- Load model and columns with error handling ----------
@st.cache_data
def load_model():
    try:
        model = joblib.load("customer_response_model.pkl")
        columns = joblib.load("columns.pkl")
        return model, columns
    except FileNotFoundError:
        st.error("❌ **Model files missing!** Please place `customer_response_model.pkl` and `columns.pkl` in the same folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Model loading error:** {e}")
        st.stop()

model, columns = load_model()
st.sidebar.success("✅ Model loaded successfully!")

# ---------- Sidebar Inputs ----------
st.sidebar.header("🎯 Customer Profile")
year_birth = st.sidebar.number_input("Year of Birth", min_value=1900, max_value=2025, value=1965)
income = st.sidebar.number_input("Income ($)", min_value=0, value=80000, format="%d")
recency = st.sidebar.slider("Recency (days since last purchase)", 0, 100, 20)

st.sidebar.subheader("💰 Spending (Monthly)")
col1, col2 = st.sidebar.columns(2)
mnt_wines = col1.number_input("Wines", min_value=0, value=1000)
mnt_meat = col2.number_input("Meat", min_value=0, value=500)
mnt_fruits = st.sidebar.number_input("Fruits", min_value=0, value=100)
mnt_fish = st.sidebar.number_input("Fish", min_value=0, value=100)
mnt_sweets = st.sidebar.number_input("Sweets", min_value=0, value=100)
mnt_gold = st.sidebar.number_input("Gold", min_value=0, value=50)

st.sidebar.subheader("🛒 Purchase Behavior")
num_deals = st.sidebar.number_input("Deals Purchases", min_value=0, value=2)
num_web = st.sidebar.number_input("Web Purchases", min_value=0, value=6)
num_catalog = st.sidebar.number_input("Catalog Purchases", min_value=0, value=4)
num_store = st.sidebar.number_input("Store Purchases", min_value=0, value=4)
num_web_visits = st.sidebar.number_input("Web Visits/Month", min_value=0, value=4)

st.sidebar.subheader("👨‍👩‍👧‍👦 Household")
kid_home = st.sidebar.number_input("Kids at Home", min_value=0, max_value=3, value=0)
teen_home = st.sidebar.number_input("Teens at Home", min_value=0, max_value=3, value=0)

st.sidebar.subheader("📚 Demographics")
education = st.sidebar.selectbox("Education", ["Graduation", "PhD", "Master", "Basic"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Together", "Divorced", "Widow", "Alone", "YOLO"])

# ---------- Prepare Input Data (COMPLETE feature engineering) ----------
@st.cache_data
def prepare_input(year_birth, income, recency, mnt_wines, mnt_fruits, mnt_meat, mnt_fish, 
                  mnt_sweets, mnt_gold, num_deals, num_web, num_catalog, num_store, 
                  num_web_visits, kid_home, teen_home, education, marital_status, columns):
    
    input_data = pd.DataFrame(0.0, index=[0], columns=columns)
    
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
    
    # Engineered feature (if in columns)
    total_spend = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweets + mnt_gold
    if 'Total_Spending' in input_data.columns:
        input_data['Total_Spending'] = total_spend
    
    # Categorical - set ALL dummies properly
    education_options = ["Basic", "Graduation", "Master", "PhD"]
    marital_options = ["Single", "Married", "Divorced", "Together", "Widow", "Alone", "YOLO"]
    
    for edu in education_options:
        col = f'Education_{edu}'
        if col in input_data.columns:
            input_data[col] = 1 if education == edu else 0
    
    for status in marital_options:
        col = f'Marital_Status_{status}'
        if col in input_data.columns:
            input_data[col] = 1 if marital_status == status else 0
    
    return input_data

# ---------- Prediction ----------
if st.button("🔮 Predict Response", type="primary", use_container_width=True):
    with st.spinner("Predicting..."):
        input_data = prepare_input(year_birth, income, recency, mnt_wines, mnt_fruits, mnt_meat, 
                                 mnt_fish, mnt_sweets, mnt_gold, num_deals, num_web, 
                                 num_catalog, num_store, num_web_visits, kid_home, 
                                 teen_home, education, marital_status, columns)
        
        try:
            prediction = model.predict(input_data)[0]
            prob_yes = model.predict_proba(input_data)[0][1]
            
            # ---------- Dashboard Layout ----------
            st.title("📊 Customer Response Dashboard")
            
            # Top metrics
            col1, col2, col3 = st.columns([2, 1, 1])
            if prediction == 1:
                col1.metric("Prediction", "✅ YES - Will Respond", delta="Great customer!")
                col2.metric("Probability", f"{prob_yes:.1%}")
                col3.success(f"Confidence: {prob_yes:.1%}")
            else:
                col1.metric("Prediction", "❌ NO - Won't Respond", delta="Try different profile")
                col2.metric("Probability", f"{prob_yes:.1%}")
                col3.warning(f"Confidence: {prob_yes:.1%}")
            
            # Key insights
            st.markdown("---")
            
            age = 2026 - year_birth
            total_spend = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweets + mnt_gold
            
            col4, col5, col6 = st.columns(3)
            col4.metric("👤 Age", f"{age} years")
            col5.metric("💵 Total Spend", f"${total_spend:,.0f}")
            col6.metric("🏠 Household", f"{kid_home} kids, {teen_home} teens")
            
            # Charts
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Spending Pie
                spending = {
                    "Wines": mnt_wines, "Meat": mnt_meat, "Fruits": mnt_fruits,
                    "Fish": mnt_fish, "Sweets": mnt_sweets, "Gold": mnt_gold
                }
                df_spending = pd.DataFrame(list(spending.items()), columns=["Category", "Amount"])
                fig1 = px.pie(df_spending, names='Category', values='Amount', 
                            title="💰 Spending Breakdown", hole=0.4)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_right:
                # Purchase Bar
                purchases = {
                    "Deals": num_deals, "Web": num_web, "Catalog": num_catalog,
                    "Store": num_store, "Web Visits": num_web_visits
                }
                df_purchases = pd.DataFrame(list(purchases.items()), columns=["Type", "Count"])
                fig2 = px.bar(df_purchases, x='Type', y='Count', color='Count',
                            title="🛒 Purchase Channels")
                st.plotly_chart(fig2, use_container_width=True)
            
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("Check if all model features match training data.")

# Show current input summary
with st.sidebar.expander("📋 Current Inputs"):
    st.json({
        "Age": 2026 - year_birth,
        "Income": f"${income:,}",
        "Total Spend": f"${mnt_wines+mnt_fruits+mnt_meat+mnt_fish+mnt_sweets+mnt_gold:,}",
        "Recency": recency,
        "Education": education,
        "Marital": marital_status
    })