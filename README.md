# Customer Churn Prediction App

A simple web app to predict whether a telecom customer is likely to churn, built with Streamlit and scikit-learn. Enter customer details through a form and get an instant prediction from either a Random Forest or Logistic Regression model.

## Demo

Fill in the customer's details (contract type, tenure, billing info, services used, etc.) and the app tells you whether they're likely to stay or leave, along with the churn probability.

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas / numpy

## Project Structure

```
├── app.py                          # Streamlit app
├── random_forest_model.pkl         # Trained Random Forest classifier
├── logistic_regression_model.pkl   # Trained Logistic Regression classifier
├── scaler.pkl                      # StandardScaler used during training
├── features.pkl                    # Feature order expected by the models
└── requirements.txt
```

## Getting Started

Clone the repo:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## How It Works

1. User fills out the form with customer attributes.
2. Categorical inputs are encoded the same way they were during model training.
3. Numerical features are scaled using the saved `StandardScaler`.
4. The selected model (Random Forest or Logistic Regression) predicts churn and returns a probability score.

## Dataset

Trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which contains customer account, billing, and service usage information along with whether they churned.

## Notes

- All four `.pkl` files must stay in the same directory as `app.py`.
- The scaler and feature order are tied to how the original models were trained, so don't swap them out without retraining.

## License

MIT
