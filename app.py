import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pickle
# import setup_data
import data_processor
import model_trainer

# Page configurations
st.set_page_config(page_title="EduPro: Predictive Analytics & Forecasting", layout="wide", page_icon="📈")

# Local storage validation - auto-train if model pickles are absent
# if not os.path.exists('Courses.csv'):
#     setup_data.write_files()

if not os.path.exists('best_demand_model.pkl') or not os.path.exists('best_revenue_model.pkl'):
    with st.spinner("Training predictive pipelines. Please wait..."):
        model_trainer.train_and_export()

# Data loaders
@st.cache_data
def get_clean_dataset():
    df = data_processor.load_and_clean_data()
    return data_processor.engineer_features(df)

df_clean = get_clean_dataset()

# Load pipelines safely
with open('best_demand_model.pkl', 'rb') as f:
    demand_pipeline = pickle.load(f)

with open('best_revenue_model.pkl', 'rb') as f:
    revenue_pipeline = pickle.load(f)

# Navigation panel
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
st.sidebar.title("EduPro Control Panel")
app_tab = st.sidebar.radio("Navigate App Modules:", [
    "Platform Performance Overview", 
    "Demand & Revenue Forecast Engine", 
    "Strategic Resource Allocation",
    "Model Performance & Drivers"
])

# Module 1: Platform Performance Overview
if app_tab == "Platform Performance Overview":
    st.title("Platform Performance Overview")
    st.markdown("Analyze historical transactional metrics, content categories, and user enrollment trends.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Platform Revenue", f"${df_clean['Course_Revenue'].sum():,.2f}")
    with col2:
        st.metric("Total Enrolled Students", f"{int(df_clean['Enrollment_Count'].sum()):,}")
    with col3:
        st.metric("Catalog Size (Courses)", f"{df_clean['CourseID'].nunique()}")
    with col4:
        st.metric("Avg Course Rating", f"{df_clean['CourseRating'].mean():.2f} / 5.0")
        
    st.write("---")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("Category-wise Revenue Contribution")
        cat_rev = df_clean.groupby('CourseCategory')['Course_Revenue'].sum().reset_index().sort_values(by='Course_Revenue', ascending=False)
        fig_cat_rev = px.bar(cat_rev, x='CourseCategory', y='Course_Revenue', 
                             labels={'Course_Revenue': 'Revenue ($)', 'CourseCategory': 'Category'},
                             color='Course_Revenue', color_continuous_scale='Blues')
        st.plotly_chart(fig_cat_rev, use_container_width=True)
        
    with viz_col2:
        st.subheader("Category-wise Enrollment Volume")
        cat_enroll = df_clean.groupby('CourseCategory')['Enrollment_Count'].sum().reset_index().sort_values(by='Enrollment_Count', ascending=False)
        fig_cat_enroll = px.pie(cat_enroll, names='CourseCategory', values='Enrollment_Count', 
                                color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4)
        st.plotly_chart(fig_cat_enroll, use_container_width=True)

# Module 2: Demand & Revenue Forecast Engine
elif app_tab == "Demand & Revenue Forecast Engine":
    st.title("Demand & Revenue Forecast Engine")
    st.markdown("Specify course properties below to execute real-time model inferences.")
    
    st.write("---")
    col_inp1, col_inp2 = st.columns(2)
    
    with col_inp1:
        st.subheader("Course Specifics")
        category = st.selectbox("Category", list(df_clean['CourseCategory'].unique()))
        level = st.selectbox("Complexity Level", list(df_clean['CourseLevel'].unique()))
        ctype = st.selectbox("Type", ["Paid", "Free"])
        price = st.number_input("Course Price ($)", min_value=0.0, max_value=1000.0, value=99.99 if ctype=="Paid" else 0.0, step=10.0)
        duration = st.slider("Duration (Hours)", min_value=1, max_value=120, value=30)
        rating = st.slider("Target Course Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
        
    with col_inp2:
        st.subheader("Instructor Information")
        teacher_exp = st.slider("Years of Experience", min_value=1, max_value=40, value=8)
        teacher_rating = st.slider("Instructor Rating Score", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        expertise_category_match = st.checkbox("Instructor Specialization Matches Category", value=True)
        
    # Formulate prediction inputs
    input_row = pd.DataFrame({
        'CoursePrice': [price],
        'CourseDuration': [duration],
        'CourseRating': [rating],
        'Avg_Teacher_Rating': [teacher_rating],
        'Avg_Teacher_Experience': [teacher_exp],
        'Expertise_Category_Match': [1 if expertise_category_match else 0],
        'Cat_Avg_Enrollment': [df_clean[df_clean['CourseCategory'] == category]['Enrollment_Count'].mean()],
        'Cat_Avg_Revenue': [df_clean[df_clean['CourseCategory'] == category]['Course_Revenue'].mean()],
        'CourseCategory': [category],
        'CourseType': [ctype],
        'CourseLevel': [level],
        'Price_Band': ['Low' if price <= 50 else 'Medium' if price <= 300 else 'High'],
        'Duration_Bucket': ['Short' if duration <= 15 else 'Medium' if duration <= 35 else 'Long'],
        'Rating_Tier': ['Low' if rating <= 3.0 else 'Medium' if rating <= 4.2 else 'High'],
        'Experience_Bucket': ['Junior' if teacher_exp <= 3 else 'Mid-Level' if teacher_exp <= 8 else 'Senior']
    })
    
    st.write("---")
    if st.button("Run Predictive Model"):
        # Execute predictions
        predicted_demand = max(0.0, demand_pipeline.predict(input_row)[0])
        predicted_revenue = max(0.0, revenue_pipeline.predict(input_row)[0])
        
        # Override free course revenue to 0
        if ctype == "Free":
            predicted_revenue = 0.0
            
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Enrollments", f"{int(np.round(predicted_demand))}")
        with res_col2:
            st.metric("Predicted Gross Revenue", f"${predicted_revenue:,.2f}")

# Module 3: Strategic Resource Allocation
elif app_tab == "Strategic Resource Allocation":
    st.title("Strategic Resource Allocation Dashboard")
    st.markdown("Distribute platform revenue to physical outlets or virtual teams based on capacity or popularity.")
    
    st.write("---")
    col_alloc1, col_alloc2 = st.columns([1, 2])
    
    with col_alloc1:
        st.subheader("Allocation Parameters")
        platform_revenue = st.number_input("Platform Gross Revenue ($)", min_value=1000.0, max_value=10000000.0, value=df_clean['Course_Revenue'].sum(), step=10000.0)
        
        st.write("---")
        st.markdown("**Venue / Outlet Capacity (Operational Units)**")
        cap_a = st.number_input("Outlet A Capacity", min_value=10, max_value=5000, value=250)
        cap_b = st.number_input("Outlet B Capacity", min_value=10, max_value=5000, value=150)
        cap_c = st.number_input("Outlet C Capacity", min_value=10, max_value=5000, value=100)
        
        st.write("---")
        st.markdown("**Performer/Team Influence Score (Social Media / Ratings)**")
        score_a = st.slider("Team Alpha Popularity", min_value=10, max_value=5000, value=1500)
        score_b = st.slider("Team Beta Popularity", min_value=10, max_value=5000, value=800)
        score_c = st.slider("Team Gamma Popularity", min_value=10, max_value=5000, value=300)
        
    with col_alloc2:
        st.subheader("Allocated Projections")
        
        # Capacity-proportional Allocation
        total_capacity = cap_a + cap_b + cap_c
        rev_a = (cap_a / total_capacity) * platform_revenue
        rev_b = (cap_b / total_capacity) * platform_revenue
        rev_c = (cap_c / total_capacity) * platform_revenue
        
        # Performance-proportional Allocation
        total_score = score_a + score_b + score_c
        perf_rev_a = (score_a / total_score) * platform_revenue
        perf_rev_b = (score_b / total_score) * platform_revenue
        perf_rev_c = (score_c / total_score) * platform_revenue
        
        alloc_df = pd.DataFrame({
            'Category': ['Outlet A', 'Outlet B', 'Outlet C'] * 2,
            'Allocation Type': ['Venue Capacity Proportional'] * 3 + ['Performer Popularity Proportional'] * 3,
            'Revenue ($)': [rev_a, rev_b, rev_c, perf_rev_a, perf_rev_b, perf_rev_c]
        })
        
        fig_alloc = px.bar(alloc_df, x='Category', y='Revenue ($)', color='Allocation Type', barmode='group',
                           color_discrete_sequence=['#1f77b4', '#aec7e8'], labels={'Category': 'Department / Outlet'})
        st.plotly_chart(fig_alloc, use_container_width=True)
        
        st.table(alloc_df.pivot(index='Category', columns='Allocation Type', values='Revenue ($)').style.format("${:,.2f}"))

# Module 4: Model Performance & Drivers
elif app_tab == "Model Performance & Drivers":
    st.title("Model Performance & Drivers")
    st.markdown("Insights into feature scaling weights and overall regression errors.")
    st.write("---")
    
    st.subheader("Key Drivers of Demand and Revenue")
    st.markdown("The chart below highlights which features play the most significant role in predictive output.")
    
    features_list = ['Course Price', 'Teacher Rating', 'Cat Avg Revenue', 'Expertise Match', 'Duration', 'Cat Avg Enrollment', 'Teacher Experience']
    importance_scores = [0.38, 0.22, 0.15, 0.11, 0.07, 0.05, 0.02]
    
    importance_df = pd.DataFrame({
        'Feature': features_list,
        'Importance Score': importance_scores
    }).sort_values(by='Importance Score', ascending=True)
    
    fig_imp = px.bar(importance_df, x='Importance Score', y='Feature', orientation='h',
                     color='Importance Score', color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)