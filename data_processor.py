import pandas as pd
import numpy as np

def load_and_clean_data(courses_path='Courses.csv', teachers_path='Teachers.csv', transactions_path='Transactions.csv'):
    """
    Loads raw CSV files, trims columns, handles type casting, aggregates transactions 
    to the course level, and outputs a merged dataframe.
    """
    courses = pd.read_csv(courses_path)
    teachers = pd.read_csv(teachers_path)
    transactions = pd.read_csv(transactions_path)
    
    # Strip whitespace from columns to avoid parsing issues
    courses.columns = courses.columns.str.strip()
    teachers.columns = teachers.columns.str.strip()
    transactions.columns = transactions.columns.str.strip()
    
    # Cast variables and handle missing entries
    courses['CoursePrice'] = pd.to_numeric(courses['CoursePrice'], errors='coerce').fillna(0.0)
    courses['CourseDuration'] = pd.to_numeric(courses['CourseDuration'], errors='coerce').fillna(courses['CourseDuration'].median())
    courses['CourseRating'] = pd.to_numeric(courses['CourseRating'], errors='coerce').fillna(courses['CourseRating'].median())
    
    teachers['YearsOfExperience'] = pd.to_numeric(teachers['YearsOfExperience'], errors='coerce').fillna(teachers['YearsOfExperience'].median())
    teachers['TeacherRating'] = pd.to_numeric(teachers['TeacherRating'], errors='coerce').fillna(teachers['TeacherRating'].median())
    teachers['Age'] = pd.to_numeric(teachers['Age'], errors='coerce').fillna(teachers['Age'].median())
    
    transactions['Amount'] = pd.to_numeric(transactions['Amount'], errors='coerce').fillna(0.0)
    
    # Merge Transactions with Teachers to link instructor details
    tx_teacher = transactions.merge(teachers, on='TeacherID', how='left')
    
    # Aggregate transaction details at the Course level
    course_stats = tx_teacher.groupby('CourseID').agg(
        Enrollment_Count=('TransactionID', 'count'),
        Course_Revenue=('Amount', 'sum'),
        Avg_Teacher_Experience=('YearsOfExperience', 'mean'),
        Avg_Teacher_Rating=('TeacherRating', 'mean'),
        Avg_Teacher_Age=('Age', 'mean'),
        Teacher_Expertise=('Expertise', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()
    
    # Merge with Course metadata
    df_merged = courses.merge(course_stats, on='CourseID', how='left')
    
    # Fill missing values for courses with no transaction history
    df_merged['Enrollment_Count'] = df_merged['Enrollment_Count'].fillna(0)
    df_merged['Course_Revenue'] = df_merged['Course_Revenue'].fillna(0.0)
    df_merged['Avg_Teacher_Experience'] = df_merged['Avg_Teacher_Experience'].fillna(teachers['YearsOfExperience'].median())
    df_merged['Avg_Teacher_Rating'] = df_merged['Avg_Teacher_Rating'].fillna(teachers['TeacherRating'].median())
    df_merged['Avg_Teacher_Age'] = df_merged['Avg_Teacher_Age'].fillna(teachers['Age'].median())
    df_merged['Teacher_Expertise'] = df_merged['Teacher_Expertise'].fillna('Unknown')
    
    return df_merged

def engineer_features(df):
    """
    Applies feature engineering transformation to numeric and categorical descriptors.
    """
    df = df.copy()
    
    # 1. Price Bands
    df['Price_Band'] = pd.cut(df['CoursePrice'], bins=[-1, 50, 300, float('inf')], labels=['Low', 'Medium', 'High'])
    
    # 2. Duration Buckets
    df['Duration_Bucket'] = pd.cut(df['CourseDuration'], bins=[-1, 15, 35, float('inf')], labels=['Short', 'Medium', 'Long'])
    
    # 3. Rating Tiers
    df['Rating_Tier'] = pd.cut(df['CourseRating'], bins=[-1, 3.0, 4.2, 5.1], labels=['Low', 'Medium', 'High'])
    
    # 4. Expertise Match score
    df['Expertise_Category_Match'] = (df['Teacher_Expertise'].str.lower().str.strip() == df['CourseCategory'].str.lower().str.strip()).astype(int)
    
    # 5. Teacher Experience Buckets
    df['Experience_Bucket'] = pd.cut(df['Avg_Teacher_Experience'], bins=[-1, 3, 8, float('inf')], labels=['Junior', 'Mid-Level', 'Senior'])
    
    # 6. Category-level aggregates to prevent target leakage
    cat_stats = df.groupby('CourseCategory').agg(
        Cat_Avg_Enrollment=('Enrollment_Count', 'mean'),
        Cat_Avg_Revenue=('Course_Revenue', 'mean')
    ).reset_index()
    
    df = df.merge(cat_stats, on='CourseCategory', how='left')
    
    # 7. Revenue per enrollment metric (Now consistently using Enrollment_Count)
    df['Revenue_Per_Enrollment'] = df.apply(
        lambda row: row['Course_Revenue'] / row['Enrollment_Count'] if row['Enrollment_Count'] > 0 else 0.0, axis=1
    )
    
    return df