from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "xgboost_model.pkl"
model = joblib.load(MODEL_PATH)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "xgboost_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)


# ============================================================
# HTML + CSS
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Insurance Cost Predictor</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, Helvetica, sans-serif;
        }

        body {
            min-height: 100vh;
            background:
                linear-gradient(135deg, #0f172a, #1e3a8a, #2563eb);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
        }

        .main-container {
            width: 100%;
            max-width: 1050px;
            background: rgba(255,255,255,0.96);
            border-radius: 25px;
            overflow: hidden;
            box-shadow: 0 25px 60px rgba(0,0,0,0.30);
        }

        /* HEADER */

        .header {
            padding: 35px 45px;
            background: linear-gradient(135deg, #2563eb, #4f46e5);
            color: white;
        }

        .header h1 {
            font-size: 34px;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 16px;
            opacity: 0.9;
        }

        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 40px;
        }

        /* LEFT INFORMATION */

        .info-section {
            background: linear-gradient(145deg, #eff6ff, #dbeafe);
            border-radius: 20px;
            padding: 30px;
        }

        .info-section h2 {
            color: #1e3a8a;
            margin-bottom: 15px;
            font-size: 25px;
        }

        .info-section p {
            color: #475569;
            line-height: 1.7;
            margin-bottom: 25px;
        }

        .feature {
            display: flex;
            align-items: center;
            margin: 18px 0;
            padding: 15px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }

        .feature-icon {
            width: 45px;
            height: 45px;
            border-radius: 12px;
            background: #2563eb;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 20px;
            margin-right: 15px;
        }

        .feature-text h3 {
            color: #1e293b;
            font-size: 15px;
            margin-bottom: 4px;
        }

        .feature-text span {
            color: #64748b;
            font-size: 13px;
        }

        /* FORM */

        .form-section {
            padding: 5px;
        }

        .form-section h2 {
            color: #0f172a;
            margin-bottom: 25px;
            font-size: 25px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 14px;
            font-weight: bold;
            color: #334155;
            margin-bottom: 8px;
        }

        .input-group input,
        .input-group select {
            padding: 13px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            outline: none;
            font-size: 14px;
            background: #f8fafc;
            transition: 0.3s;
        }

        .input-group input:focus,
        .input-group select:focus {
            border-color: #2563eb;
            background: white;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.10);
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .predict-btn {
            width: 100%;
            margin-top: 25px;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb, #4f46e5);
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
            box-shadow: 0 8px 20px rgba(37,99,235,0.30);
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 25px rgba(37,99,235,0.40);
        }

        /* RESULT */

        .result {
            margin-top: 25px;
            padding: 22px;
            border-radius: 15px;
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            text-align: center;
            border: 1px solid #86efac;
        }

        .result h3 {
            color: #166534;
            font-size: 16px;
            margin-bottom: 8px;
        }

        .result .amount {
            font-size: 32px;
            font-weight: bold;
            color: #15803d;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            background: #fee2e2;
            color: #991b1b;
            text-align: center;
        }

        .footer {
            text-align: center;
            padding: 18px;
            color: #64748b;
            font-size: 13px;
            border-top: 1px solid #e2e8f0;
        }

        @media(max-width: 800px) {

            .content {
                grid-template-columns: 1fr;
                padding: 25px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .full-width {
                grid-column: auto;
            }

            .header h1 {
                font-size: 27px;
            }
        }

    </style>
</head>

<body>

<div class="main-container">

    <!-- HEADER -->

    <div class="header">
        <h1>💳 Insurance Cost Predictor</h1>
        <p>AI-powered insurance cost prediction using XGBoost</p>
    </div>


    <div class="content">

        <!-- INFORMATION -->

        <div class="info-section">

            <h2>Smart Prediction</h2>

            <p>
                Enter the customer's personal and insurance information.
                Our trained XGBoost machine learning model will estimate
                the expected insurance cost.
            </p>

            <div class="feature">

                <div class="feature-icon">🤖</div>

                <div class="feature-text">
                    <h3>XGBoost Model</h3>
                    <span>Machine learning powered prediction</span>
                </div>

            </div>

            <div class="feature">

                <div class="feature-icon">⚡</div>

                <div class="feature-text">
                    <h3>Fast Prediction</h3>
                    <span>Get your result instantly</span>
                </div>

            </div>

            <div class="feature">

                <div class="feature-icon">📊</div>

                <div class="feature-text">
                    <h3>6 Input Features</h3>
                    <span>Age, BMI, children, smoker & more</span>
                </div>

            </div>

            <div class="feature">

                <div class="feature-icon">🔒</div>

                <div class="feature-text">
                    <h3>Simple & Secure</h3>
                    <span>Your information is used only for prediction</span>
                </div>

            </div>

        </div>


        <!-- FORM -->

        <div class="form-section">

            <h2>Enter Details</h2>

            <form method="POST">

                <div class="form-grid">

                    <!-- AGE -->

                    <div class="input-group">

                        <label>Age</label>

                        <input
                            type="number"
                            name="age"
                            min="1"
                            max="100"
                            placeholder="Enter age"
                            required
                        >

                    </div>


                    <!-- SEX -->

                    <div class="input-group">

                        <label>Sex</label>

                        <select name="sex" required>

                            <option value="">Select sex</option>
                            <option value="0">Female</option>
                            <option value="1">Male</option>

                        </select>

                    </div>


                    <!-- BMI -->

                    <div class="input-group">

                        <label>BMI</label>

                        <input
                            type="number"
                            name="bmi"
                            step="0.01"
                            min="1"
                            max="100"
                            placeholder="e.g. 25.5"
                            required
                        >

                    </div>


                    <!-- CHILDREN -->

                    <div class="input-group">

                        <label>Children</label>

                        <input
                            type="number"
                            name="children"
                            min="0"
                            max="20"
                            placeholder="Number of children"
                            required
                        >

                    </div>


                    <!-- SMOKER -->

                    <div class="input-group">

                        <label>Smoker</label>

                        <select name="smoker" required>

                            <option value="">Select option</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>

                        </select>

                    </div>


                    <!-- REGION -->

                    <div class="input-group">

                        <label>Region</label>

                        <select name="region" required>

                            <option value="">Select region</option>
                            <option value="0">Northeast</option>
                            <option value="1">Northwest</option>
                            <option value="2">Southeast</option>
                            <option value="3">Southwest</option>

                        </select>

                    </div>

                </div>


                <button class="predict-btn" type="submit">
                    🔮 Predict Insurance Cost
                </button>

            </form>


            {% if prediction is not none %}

            <div class="result">

                <h3>Estimated Insurance Cost</h3>

                <div class="amount">
                    ₹ {{ "{:,.2f}".format(prediction) }}
                </div>

            </div>

            {% endif %}


            {% if error %}

            <div class="error">
                ⚠️ {{ error }}
            </div>

            {% endif %}

        </div>

    </div>


    <div class="footer">
        © 2026 Insurance Cost Predictor | Powered by XGBoost & Flask
    </div>

</div>

</body>
</html>
"""


# ============================================================
# ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            if not model_loaded:
                raise Exception(
                    "Model could not be loaded. Check your PKL file and dependencies."
                )

            # Get form values
            age = float(request.form["age"])
            sex = int(request.form["sex"])
            bmi = float(request.form["bmi"])
            children = int(request.form["children"])
            smoker = int(request.form["smoker"])
            region = int(request.form["region"])

            # Create dataframe with EXACT feature names
            input_data = pd.DataFrame(
                [[
                    age,
                    sex,
                    bmi,
                    children,
                    smoker,
                    region
                ]],
                columns=[
                    "Age",
                    "Sex",
                    "BMI",
                    "Children",
                    "Smoker",
                    "Region"
                ]
            )

            # Prediction
            prediction = float(model.predict(input_data)[0])

        except Exception as e:

            error = f"Prediction Error: {str(e)}"

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
```
