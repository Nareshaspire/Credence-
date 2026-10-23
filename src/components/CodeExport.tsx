import React, { useState } from 'react';
import { ModelAlgorithm, ModelMetrics } from '../types';

interface CodeExportProps {
  activeModelKey: ModelAlgorithm;
  activeMetrics: ModelMetrics;
}

export const CodeExport: React.FC<CodeExportProps> = ({
  activeModelKey,
  activeMetrics,
}) => {
  const [selectedLanguage, setSelectedLanguage] = useState<'fastapi' | 'streamlit' | 'sklearn'>(
    'fastapi'
  );
  const [copied, setCopied] = useState(false);

  // FastAPI Python Code
  const fastapiCode = `# ==============================================================================
# Credence ML — Production Loan Risk Prediction Service
# Framework: FastAPI + Scikit-Learn + Pydantic v2
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ==============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI(
    title="Loan Risk Prediction API",
    description="Underwriting default probability and risk scoring service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Data Schema
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., example=15000.0, description="Principal requested in USD")
    term_months: int = Field(36, example=36, description="Loan term in months (36 or 60)")
    annual_inc: float = Field(..., example=75000.0, description="Verified annual income")
    emp_length_years: int = Field(..., example=5, description="Years in current employment")
    home_ownership: str = Field("RENT", example="MORTGAGE", description="RENT, OWN, or MORTGAGE")
    fico_score: int = Field(..., ge=580, le=850, example=710, description="FICO bureau score")
    dti: float = Field(..., example=18.5, description="Debt-to-income ratio percentage")
    revol_util: float = Field(..., example=42.0, description="Revolving credit utilization %")
    delinq_2yrs: int = Field(0, example=0, description="Delinquencies in past 24 months")
    inq_last_6mths: int = Field(0, example=1, description="Hard inquiries in past 6 months")
    loan_purpose: str = Field("debt_consolidation", example="debt_consolidation")

# Response Data Schema
class RiskPredictionResponse(BaseModel):
    default_probability: float
    decision: str
    risk_tier: str
    recommended_apr: float
    expected_loss_usd: float

def compute_engineered_features(app: LoanApplication):
    # Installment formula
    r = 0.12 / 12  # baseline rate
    n = app.term_months
    installment = (app.loan_amnt * (r * (1 + r)**n)) / ((1 + r)**n - 1)
    monthly_inc = max(1.0, app.annual_inc / 12)
    installment_burden = (installment / monthly_inc) * 100
    credit_stability = (app.fico_score / 850) * 70 + (100 - app.revol_util) * 0.3
    risk_interaction = (app.dti * (850 - app.fico_score)) / 270
    delinq_severity = app.delinq_2yrs * 1.8 + app.inq_last_6mths * 1.1

    vector = [
        app.fico_score,
        app.dti,
        app.annual_inc,
        app.loan_amnt,
        12.0,
        app.revol_util,
        app.delinq_2yrs,
        app.inq_last_6mths,
        app.emp_length_years,
        20, # total accounts proxy
        installment_burden,
        credit_stability,
        risk_interaction,
        delinq_severity,
        1 if app.term_months == 60 else 0,
        1 if app.home_ownership == "RENT" else 0,
        1 if app.home_ownership == "OWN" else 0,
        1 if app.loan_purpose == "small_business" else 0,
    ]
    return np.array(vector).reshape(1, -1)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "${activeMetrics.algorithmName}"}

@app.post("/api/v1/predict", response_model=RiskPredictionResponse)
def predict_loan_risk(application: LoanApplication):
    try:
        # Preprocessing & Vectorization
        features = compute_engineered_features(application)
        
        # In production: model = joblib.load("model.joblib")
        # Simulating active model probability calibration:
        logit = (
            -4.0
            + (700 - application.fico_score) * 0.018
            + (application.dti - 18) * 0.062
            + application.delinq_2yrs * 0.45
            + (application.revol_util - 50) * 0.02
        )
        prob = float(1 / (1 + np.exp(-logit)))
        prob = min(0.99, max(0.01, prob))

        # Risk Classification
        if prob < 0.12:
            tier, decision, apr = "AAA", "APPROVED", 6.25
        elif prob < 0.25:
            tier, decision, apr = "A", "APPROVED", 10.50
        elif prob < 0.40:
            tier, decision, apr = "BBB", "CONDITIONAL_REVIEW", 16.20
        else:
            tier, decision, apr = "D", "DECLINED_HIGH_RISK", 26.50

        expected_loss = float(application.loan_amnt * prob * 0.70)

        return RiskPredictionResponse(
            default_probability=round(prob, 4),
            decision=decision,
            risk_tier=tier,
            recommended_apr=apr,
            expected_loss_usd=round(expected_loss, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
`;

  // Streamlit Python Code
  const streamlitCode = `# ==============================================================================
# Credence ML — Interactive Streamlit Dashboard
# Run with: streamlit run streamlit_app.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Loan Risk Assessment System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Credence Loan Risk Underwriting System")
st.markdown("Automated Credit Default Prediction Engine using Machine Learning")

# Sidebar - Applicant Inputs
st.sidebar.header("Applicant Financial Inputs")
loan_amnt = st.sidebar.slider("Loan Amount ($)", 1000, 40000, 15000, step=500)
term = st.sidebar.selectbox("Term", [36, 60], format_func=lambda x: f"{x} Months")
fico = st.sidebar.slider("FICO Score", 580, 850, 710)
income = st.sidebar.number_input("Annual Income ($)", min_value=15000, max_value=300000, value=75000, step=2500)
dti = st.sidebar.slider("Debt-to-Income (DTI %)", 4.0, 45.0, 18.5, step=0.5)
revol_util = st.sidebar.slider("Revolving Line Utilization (%)", 0.0, 100.0, 42.0)
delinq = st.sidebar.number_input("Delinquencies (24m)", min_value=0, max_value=5, value=0)
inq = st.sidebar.number_input("Recent Inquiries (6m)", min_value=0, max_value=5, value=1)
purpose = st.sidebar.selectbox("Purpose", ["debt_consolidation", "credit_card", "small_business", "home_improvement"])

# Prediction Calculation
logit = (
    -4.0
    + (700 - fico) * 0.018
    + (dti - 18) * 0.062
    + delinq * 0.45
    + (revol_util - 50) * 0.02
)
default_prob = float(1 / (1 + np.exp(-logit)))

# Results Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted Default Risk", f"{default_prob * 100:.1f}%")

with col2:
    if default_prob < 0.20:
        st.metric("Underwriting Disposition", "APPROVED", delta="Low Risk", delta_color="normal")
    elif default_prob < 0.40:
        st.metric("Underwriting Disposition", "CONDITIONAL", delta="Review Required", delta_color="off")
    else:
        st.metric("Underwriting Disposition", "DECLINED", delta="High Risk", delta_color="inverse")

with col3:
    tier = "AAA" if default_prob < 0.08 else "A" if default_prob < 0.20 else "BBB" if default_prob < 0.35 else "D"
    st.metric("Credit Tier", tier)

with col4:
    exp_loss = loan_amnt * default_prob * 0.70
    st.metric("Projected Loss (70% LGD)", f"\${exp_loss:,.0f}")

st.divider()

# Explainability Section
st.subheader("Factor Attribution Breakdown")
st.write("How applicant variables impact baseline default likelihood:")

data = [
    {"Factor": "FICO Credit Score", "Impact": (700 - fico) * -0.055 * 100},
    {"Factor": "Debt-to-Income (DTI)", "Impact": (dti - 18) * 0.038 * 100},
    {"Factor": "Revolving Line Utilization", "Impact": (revol_util - 45) * 0.024 * 100},
    {"Factor": "Past Delinquencies", "Impact": delinq * 8.0},
]
df_impact = pd.DataFrame(data)
st.bar_chart(df_impact.set_index("Factor"))
`;

  // Scikit-Learn Python Training Code
  const sklearnCode = `# ==============================================================================
# Credence ML — Scikit-Learn Training & Evaluation Pipeline
# Libraries: pandas, numpy, scikit-learn, joblib
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, brier_score_loss
import joblib

# 1. Load Dataset
print("Loading LendingClub credit default dataset...")
df = pd.read_csv("cleaned_lending_club_dataset.csv")

# 2. Feature Engineering
print("Computing domain risk indices...")
df["monthly_income"] = df["annual_inc"] / 12
r = (df["int_rate"] / 100) / 12
n = df["term_months"]
df["installment"] = (df["loan_amnt"] * (r * (1 + r)**n)) / ((1 + r)**n - 1)
df["installment_burden"] = (df["installment"] / df["monthly_income"]) * 100
df["credit_stability_index"] = (df["fico_score"] / 850) * 70 + (100 - df["revol_util"]) * 0.3
df["risk_interaction"] = (df["dti"] * (850 - df["fico_score"])) / 270
df["delinquency_severity"] = df["delinq_2yrs"] * 1.8 + df["inq_last_6mths"] * 1.1

# One-hot encoding
df = pd.get_dummies(df, columns=["home_ownership", "loan_purpose"], drop_first=True)

feature_cols = [c for c in df.columns if c not in ["id", "default", "sub_grade", "grade"]]
X = df[feature_cols]
y = df["default"]

# 3. Stratified Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 4. Feature Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training & Comparison
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42)
}

print("\\n" + "="*50)
print("MODEL BENCHMARK RESULTS")
print("="*50)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"\\nModel: {name}")
    print(f"ROC-AUC Score: {auc:.4f} | Brier Score: {brier:.4f}")
    print(classification_report(y_test, preds, target_names=["Paid", "Default"]))

# 6. Save Best Artifact
best_model = models["Gradient Boosting"]
joblib.dump({"model": best_model, "scaler": scaler, "features": feature_cols}, "loan_risk_model.joblib")
print("\\nSuccessfully saved trained pipeline to loan_risk_model.joblib!")
`;

  const getActiveCode = () => {
    switch (selectedLanguage) {
      case 'streamlit':
        return streamlitCode;
      case 'sklearn':
        return sklearnCode;
      case 'fastapi':
      default:
        return fastapiCode;
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(getActiveCode());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadFile = () => {
    const filename =
      selectedLanguage === 'fastapi'
        ? 'main.py'
        : selectedLanguage === 'streamlit'
        ? 'streamlit_app.py'
        : 'train_model.py';

    const blob = new Blob([getActiveCode()], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Production Deployment & Code Export Hub
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
            <span>FastAPI Microservice</span>
            <span aria-hidden="true">·</span>
            <span>Streamlit Interactive App</span>
            <span aria-hidden="true">·</span>
            <span>Scikit-Learn Training Pipeline</span>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            onClick={copyToClipboard}
            className="px-3.5 py-2 text-xs font-semibold text-slate-200 bg-slate-800 border border-slate-700 rounded-lg hover:bg-slate-700 transition-colors whitespace-nowrap"
          >
            {copied ? 'Copied to Clipboard!' : 'Copy Code'}
          </button>
          <button
            onClick={downloadFile}
            className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-500 transition-colors whitespace-nowrap"
          >
            Download .py File
          </button>
        </div>
      </div>

      {/* Code Language Switcher */}
      <div className="p-1 bg-slate-900 border border-slate-800 rounded-xl flex items-center gap-1">
        <button
          onClick={() => setSelectedLanguage('fastapi')}
          className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors ${
            selectedLanguage === 'fastapi'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          FastAPI REST Microservice (main.py)
        </button>
        <button
          onClick={() => setSelectedLanguage('streamlit')}
          className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors ${
            selectedLanguage === 'streamlit'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Streamlit Web Dashboard (streamlit_app.py)
        </button>
        <button
          onClick={() => setSelectedLanguage('sklearn')}
          className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors ${
            selectedLanguage === 'sklearn'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Scikit-Learn Training Script (train_model.py)
        </button>
      </div>

      {/* Code Editor Window */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
        <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900 border-b border-slate-800 text-xs">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-700" />
            <span className="w-2.5 h-2.5 rounded-full bg-slate-700" />
            <span className="w-2.5 h-2.5 rounded-full bg-slate-700" />
            <span className="font-mono text-slate-300 ml-2">
              {selectedLanguage === 'fastapi'
                ? 'main.py (FastAPI REST Service)'
                : selectedLanguage === 'streamlit'
                ? 'streamlit_app.py (Streamlit UI)'
                : 'train_model.py (Scikit-Learn)'}
            </span>
          </div>
          <span className="text-3xs font-mono text-slate-500">Python 3.10+ / Scikit-learn 1.4+</span>
        </div>

        <pre className="p-5 font-mono text-xs text-slate-300 overflow-x-auto leading-relaxed max-h-[580px] overflow-y-auto">
          {getActiveCode()}
        </pre>
      </div>
    </div>
  );
};
