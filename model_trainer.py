import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_processor import load_and_clean_data, engineer_features

def build_preprocessing_pipeline(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor

def train_and_export():
    df = load_and_clean_data()
    df = engineer_features(df)
    
    numeric_features = ['CoursePrice', 'CourseDuration', 'CourseRating', 'Avg_Teacher_Rating', 'Avg_Teacher_Experience', 'Expertise_Category_Match', 'Cat_Avg_Enrollment', 'Cat_Avg_Revenue']
    categorical_features = ['CourseCategory', 'CourseType', 'CourseLevel', 'Price_Band', 'Duration_Bucket', 'Rating_Tier', 'Experience_Bucket']
    
    X = df[numeric_features + categorical_features]
    y_demand = df['Enrollment_Count']
    y_revenue = df['Course_Revenue']
    
    X_train, X_test, y_train_d, y_test_d = train_test_split(X, y_demand, test_size=0.2, random_state=42)
    _, _, y_train_r, y_test_r = train_test_split(X, y_revenue, test_size=0.2, random_state=42)
    
    candidate_models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    metrics_report = {}
    best_pipelines = {}
    
    for target_name, (y_tr, y_te) in {'Demand': (y_train_d, y_test_d), 'Revenue': (y_train_r, y_test_r)}.items():
        metrics_report[target_name] = {}
        top_r2 = -float('inf')
        selected_model_name = None
        
        for name, model in candidate_models.items():
            pipeline = Pipeline(steps=[
                ('processor', build_preprocessing_pipeline(categorical_features, numeric_features)),
                ('model', model)
            ])
            pipeline.fit(X_train, y_tr)
            preds = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_te, preds)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            r2 = r2_score(y_te, preds)
            
            metrics_report[target_name][name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            if r2 > top_r2:
                top_r2 = r2
                selected_model_name = name
        
        # Fit optimal model to total dataset
        optimal_pipeline = Pipeline(steps=[
            ('processor', build_preprocessing_pipeline(categorical_features, numeric_features)),
            ('model', candidate_models[selected_model_name])
        ])
        
        if target_name == 'Demand':
            optimal_pipeline.fit(X, y_demand)
        else:
            optimal_pipeline.fit(X, y_revenue)
            
        best_pipelines[target_name] = optimal_pipeline
        
        # Save model objects
        with open(f'best_{target_name.lower()}_model.pkl', 'wb') as f:
            pickle.dump(optimal_pipeline, f)
            
    print("Models successfully trained and exported.")
    return metrics_report

if __name__ == '__main__':
    train_and_export()