📊 Course Demand and Revenue Forecasting

A Machine Learning-powered web application that forecasts course demand and estimated revenue by analyzing historical educational platform data. The system leverages trained machine learning models to provide accurate predictions, enabling educational institutions and EdTech companies to make informed strategic and business decisions.

🚀 Overview

The Course Demand and Revenue Forecasting project is built to examine historical information related to courses, teachers, users, and transactions. By processing this data, the application predicts future enrollment demand along with expected revenue, helping organizations optimize planning and resource allocation.

The application provides predictions for:

📈 Future Course Demand
💰 Estimated Revenue

Along with prediction capabilities, the project also includes complete data preprocessing, model training pipelines, pre-trained machine learning models, and an interactive Streamlit interface for users.

✨ Features

The application offers several key functionalities, including:

📊 Data cleaning and preprocessing
🤖 Machine learning model training
📈 Future course demand prediction
💰 Revenue forecasting
💾 Support for pre-trained .pkl models
🌐 Interactive web interface built with Streamlit
📁 CSV dataset integration
🛠️ Tech Stack
Category	Technology
Language	Python
Framework	Streamlit
Data Analysis	Pandas, NumPy
Machine Learning	Scikit-learn
Model Serialization	Pickle
Visualization	Matplotlib, Seaborn
📂 Project Structure
Course-Demand-and-Revenue-Forecasting/
│
├── app.py                     # Streamlit web application
├── data_processor.py          # Data preprocessing and feature engineering
├── model_trainer.py           # Model training script
│
├── Courses.csv                # Course dataset
├── Teachers.csv               # Teacher dataset
├── Users.csv                  # User dataset
├── Transactions.csv           # Transaction dataset
│
├── best_demand_model.pkl      # Trained demand prediction model
├── best_revenue_model.pkl     # Trained revenue prediction model
│
├── requirements.txt           # Required Python libraries
├── README.md                  # Project documentation
└── .gitignore                 # Git ignored files
⚙️ Installation

Follow the steps below to set up the project on your local machine.

Clone the Repository
git clone https://github.com/DeepGhosh46/Course-Demand-and-Revenue-Forecasting.git
Navigate to the Project Directory
cd Course-Demand-and-Revenue-Forecasting
Create a Virtual Environment (Optional)
python -m venv venv

Activate the virtual environment based on your operating system.

Windows

venv\Scripts\activate

Linux/macOS

source venv/bin/activate
Install Required Dependencies
pip install -r requirements.txt
▶️ Running the Application

Launch the Streamlit application using the following command:

streamlit run app.py

Once the application starts, Streamlit will display a local URL (typically http://localhost:8501). Open this address in your browser to access the web application.

📊 Machine Learning Workflow

The complete prediction pipeline follows these steps:

Load the CSV datasets.
Clean and preprocess the data.
Perform feature engineering.
Train the forecasting models.
Save the trained models using Pickle.
Load the saved models within the Streamlit application.
Generate demand and revenue predictions.
📁 Dataset

The project relies on four separate datasets, each serving a specific purpose during preprocessing and model training.

File	Description
Courses.csv	Course information
Teachers.csv	Teacher details
Users.csv	Student/User information
Transactions.csv	Enrollment and payment history
📈 Trained Models

Two machine learning models are included with the project for prediction tasks.

best_demand_model.pkl → Predicts future course demand.
best_revenue_model.pkl → Predicts expected revenue.
📌 Requirements

Install all required libraries by running:

pip install -r requirements.txt

The project primarily depends on the following packages:

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
🔮 Future Improvements

The project can be enhanced further with several additional features, including:

📊 Interactive analytics dashboards
☁️ Cloud deployment support
📈 Advanced forecasting algorithms such as XGBoost and LightGBM
🧠 Deep Learning models (LSTM)
🔗 Database integration with MySQL or PostgreSQL
📤 Exportable prediction reports
🤝 Contributing

Contributions are always welcome.

To contribute:

Fork this repository.
Create a new feature branch.
git checkout -b feature-name
Commit your changes.
git commit -m "Add new feature"
Push the branch to your GitHub repository.
git push origin feature-name
Open a Pull Request for review.
📜 License

This project has been developed for educational and research purposes.

👨‍💻 Author

Deep Ghosh

GitHub: https://github.com/DeepGhosh46

⭐ Support

If you found this project useful, consider giving it a ⭐ Star on GitHub. Your support helps encourage continued development, improvements, and future updates.
