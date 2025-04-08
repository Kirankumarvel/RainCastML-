
# 🌧️ RainCastML – Smarter Forecasts with Data-Driven Precision

RainCastML is a machine learning pipeline that predicts whether it will rain today in the Melbourne area using weather data. It applies preprocessing, model tuning, and performance evaluation using Scikit-learn’s powerful tools like Pipelines and GridSearchCV.

## 🚀 Features

- 📊 Automatic detection of numerical & categorical features
- 🧼 Preprocessing pipeline with scaling and one-hot encoding
- 🌲 Random Forest & Logistic Regression classifiers with hyperparameter tuning
- ✅ Stratified K-Fold Cross-Validation
- 📈 Performance metrics: Accuracy, Confusion Matrix, Classification Report
- 🔍 Feature Importance analysis
- 📦 Clean, modular code with extensibility in mind

---

## 🧠 Problem Statement

> Predict whether it will rain today (`RainToday`) based on historical weather attributes in Melbourne.

---

## 📁 Project Structure

```bash
RainCastML/
│
├── data/
│   └── weather_data.csv              # Input dataset
│
├── notebooks/
│   └── RainCastML_Pipeline.py       # End-to-end model pipeline
│
├── models/
│   └── best_model.pkl                # Saved model after GridSearchCV (optional)
│
├── plots/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── README.md                         # This file
└── requirements.txt                  # Required Python packages
```

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

## 📊 Model Pipeline

1. **Preprocessing**
   - Numerical: StandardScaler
   - Categorical: OneHotEncoder
2. **ColumnTransformer** to unify transformations
3. **Model Selection**
   - Random Forest
   - Logistic Regression
4. **Grid Search with Cross-Validation**
5. **Evaluation**
   - Accuracy
   - Confusion Matrix
   - Classification Report
   - Feature Importances

---

## 📈 Sample Results

```text
Best parameters found: {'classifier__max_depth': 10, 'classifier__n_estimators': 100}
Best cross-validation score: 0.84
Test set score: 0.83
```

![Feature Importances](plots/feature_importance.png)

---

## 🧪 Try It Yourself

### 1. Clone this repo

```bash
git clone https://github.com/Kirankumarvel/RainCastML-.git
cd RainCastML-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `notebooks/RainCastML_Pipeline.ipynb` in Jupyter or VSCode to explore the full pipeline.

---

## 🤖 Future Improvements

- Add XGBoost & SVM classifiers
- Use SMOTE for class imbalance
- Integrate weather API for real-time prediction
- Streamlit app deployment

---

## 📌 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🌦️ Built with ❤️ by [Kirankumarvel](https://github.com/Kirankumarvel)

