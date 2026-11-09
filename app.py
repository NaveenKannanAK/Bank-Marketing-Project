
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ==========================================================
# 1. LOAD TRAINED MODEL AND PREPROCESSING FILES
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

model = joblib.load(
    os.path.join(MODEL_DIR, "best_random_forest_model.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

threshold = joblib.load(
    os.path.join(MODEL_DIR, "classification_threshold.pkl")
)

# Numerical columns used during model training
numerical_columns = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]


# ==========================================================
# 2. HOME PAGE
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# 3. CUSTOMER SUBSCRIPTION PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Collect customer information
        input_data = {
            "age": float(request.form["age"]),
            "job": request.form["job"],
            "marital": request.form["marital"],
            "education": request.form["education"],
            "default": request.form["default"],
            "housing": request.form["housing"],
            "loan": request.form["loan"],
            "contact": request.form["contact"],
            "month": request.form["month"],
            "day_of_week": request.form["day_of_week"],
            "campaign": float(request.form["campaign"]),
            "pdays": float(request.form["pdays"]),
            "previous": float(request.form["previous"]),
            "poutcome": request.form["poutcome"],
            "emp.var.rate": float(request.form["emp_var_rate"]),
            "cons.price.idx": float(request.form["cons_price_idx"]),
            "cons.conf.idx": float(request.form["cons_conf_idx"]),
            "euribor3m": float(request.form["euribor3m"]),
            "nr.employed": float(request.form["nr_employed"])
        }

        # Convert input into DataFrame
        df = pd.DataFrame([input_data])

        # Apply one-hot encoding
        df = pd.get_dummies(df)

        # Match the exact training feature columns and order
        df = df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scale numerical features
        df[numerical_columns] = scaler.transform(
            df[numerical_columns]
        )

        # Predict subscription probability
        probability = model.predict_proba(df)[0][1]

        # Apply saved classification threshold
        prediction = int(probability >= threshold)

        # Generate prediction result
        if prediction == 1:

            result = "Likely to Subscribe"

            recommendation = (
                "High-priority customer. "
                "The marketing team should consider "
                "a personalized term deposit offer."
            )

            status = "success"

        else:

            result = "Unlikely to Subscribe"

            recommendation = (
                "Lower-priority customer. "
                "Consider targeted follow-up campaigns "
                "and alternative marketing strategies."
            )

            status = "warning"

        # Convert probability into percentage
        confidence = round(probability * 100, 2)

        # Display prediction result
        return render_template(
            "result.html",
            result=result,
            probability=confidence,
            recommendation=recommendation,
            status=status,
            threshold=round(threshold * 100, 2)
        )

    except (ValueError, KeyError) as e:

        return f"Invalid or missing input: {str(e)}", 400

    except Exception as e:

        app.logger.exception("Prediction failed")
        return "An unexpected prediction error occurred.", 500


# ==========================================================
# 4. ABOUT PAGE
# ==========================================================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================================================
# 5. HEALTH CHECK ROUTE
# ==========================================================

@app.route("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "threshold": float(threshold)
    }, 200


# ==========================================================
# 6. RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
