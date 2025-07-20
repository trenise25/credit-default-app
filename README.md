# 💳 Credit Default Risk Predictor

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-orange?logo=streamlit)](https://credit-default-app.streamlit.app/)

A machine learning-powered Streamlit web app that predicts the probability of a customer defaulting on their credit card payment — based on financial and behavioral features. Built using Python, XGBoost, and deployed live on [Streamlit Cloud](https://credit-default-app.streamlit.app/).

---

## 📌 Project Overview

Credit default risk is a key concern for banks and financial institutions. Using the **UCI Credit Card Dataset**, this project demonstrates an end-to-end pipeline:
- Data preprocessing & analysis
- ML modeling (XGBoost)
- Risk scoring with probability
- Real-time prediction via a web app

---

## 🚀 Live Demo

👉 Try the app here: [https://credit-default-app.streamlit.app](https://credit-default-app.streamlit.app)

You can input:
- Credit limit
- Age, gender, education, marital status
- Repayment history (last 6 months)
- Monthly bill amounts and payments

The app will predict:
- ✅ Whether the customer will default
- 📊 Probability of default

---

## 📊  Screenshot

<img width="1919" height="815" alt="image" src="https://github.com/user-attachments/assets/92032a97-2848-40df-802a-bff6cc2bf387" />
<img width="1917" height="661" alt="image" src="https://github.com/user-attachments/assets/4fa66438-5b95-4073-b452-f4b0284fc5af" />
<img width="1919" height="676" alt="image" src="https://github.com/user-attachments/assets/1bf60b94-a4b2-4579-b76e-a1996b8eacf9" />
<img width="1914" height="653" alt="image" src="https://github.com/user-attachments/assets/16c9607a-2f1e-49bc-a0e6-8610dd229ebf" />
<img width="1919" height="729" alt="image" src="https://github.com/user-attachments/assets/6e30b028-d60e-4f5b-b6e8-bfae817b6957" />
<img width="1919" height="313" alt="image" src="https://github.com/user-attachments/assets/c907d9a0-6b3f-4003-bcb3-c2c4e382f37e" />



## 📁 Dataset

- **Source**: [UCI Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Records**: 30,000 clients
- **Features**: Demographics, repayment history, bill amounts, payment amounts

---

## 📈 Model Details

| Metric         | Value        |
|----------------|--------------|
| Algorithm      | XGBoost Classifier |
| Accuracy       | ~76%         |
| ROC AUC Score  | ~0.78        |
| Trained On     | 24 features  |

Model trained using:
- `scikit-learn` and `xgboost`
- Feature scaling with `StandardScaler`
- Evaluation using confusion matrix, F1-score, ROC AUC

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/trenise25/credit-default-app.git
cd credit-default-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
