# 📊 Course Demand and Revenue Forecasting

A Machine Learning-powered web application that predicts **course demand** and **estimated revenue** by analyzing historical educational platform data. The application utilizes trained machine learning models to help educational institutions and EdTech platforms make informed decisions based on historical trends.

---

## 🚀 Overview

The **Course Demand and Revenue Forecasting** project is designed to transform historical educational data into actionable business insights. By analyzing information related to courses, teachers, users, and transactions, the application estimates future course demand and expected revenue.

The project includes an end-to-end machine learning workflow, beginning with data preprocessing and feature engineering, followed by model training and evaluation. The trained models are then integrated into an interactive Streamlit application, allowing users to generate predictions quickly and efficiently without requiring technical expertise.

### Predictions Provided

- 📈 Future Course Demand
- 💰 Estimated Revenue

---

## ✨ Features

The application provides a complete forecasting pipeline with the following capabilities:

- 📊 Data preprocessing and cleaning
- 🤖 Machine Learning model training
- 📈 Course demand prediction
- 💰 Revenue forecasting
- 💾 Support for pre-trained `.pkl` models
- 🌐 Interactive Streamlit web application
- 📁 CSV-based dataset support

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Framework | Streamlit |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Pickle |
| Visualization | Matplotlib, Seaborn |

---

## 📂 Project Structure

```text
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
```

---

## ⚙️ Installation

Follow the steps below to run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/DeepGhosh46/Course-Demand-and-Revenue-Forecasting.git
```

### 2. Navigate to the Project Directory

```bash
cd Course-Demand-and-Revenue-Forecasting
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit application using:

```bash
streamlit run app.py
```

After execution, Streamlit will generate a local URL (typically **http://localhost:8501**). Open the URL in your browser to access the application.

---

## 📊 Machine Learning Workflow

The forecasting pipeline follows these stages:

1. Load CSV datasets
2. Clean and preprocess the data
3. Perform feature engineering
4. Train forecasting models
5. Save trained models using Pickle
6. Load models within the Streamlit application
7. Generate demand and revenue predictions

---

## 📁 Dataset

The project uses four datasets that represent different aspects of an educational platform.

| File | Description |
|------|-------------|
| `Courses.csv` | Course information |
| `Teachers.csv` | Teacher details |
| `Users.csv` | Student/User information |
| `Transactions.csv` | Enrollment and payment history |

---

## 📈 Trained Models

The project includes two pre-trained machine learning models:

- **best_demand_model.pkl** → Predicts future course demand.
- **best_revenue_model.pkl** → Predicts expected revenue.

---

## 📌 Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries used in the project include:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## 🔮 Future Improvements

Potential enhancements for future versions include:

- 📊 Interactive dashboards
- ☁️ Cloud deployment
- 📈 Advanced forecasting algorithms (XGBoost, LightGBM)
- 🧠 Deep Learning models (LSTM)
- 🔗 Database integration (MySQL/PostgreSQL)
- 📤 Exportable prediction reports

---

## 🤝 Contributing

Contributions are always welcome.

To contribute:

1. Fork this repository.
2. Create a new feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

## 📜 License

This project is developed for educational and research purposes.

---

## 👨‍💻 Author

**Deep Ghosh**

GitHub: https://github.com/DeepGhosh46

---

## ⭐ Support

If you found this project helpful, please consider giving it a **⭐ Star** on GitHub. Your support encourages future development, improvements, and new features.
