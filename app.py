"""Streamlit app for heart disease risk screening.  Run: streamlit run app.py"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart disease risk screening", page_icon="🫀", layout="wide")

LO, MID, HI = "#2E7D6B", "#D99A2B", "#C23B4E"
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{--ink:#12303B;--muted:#5B7480;--line:#D5E0E4}
.stApp,.stApp p,.stApp label,.stApp input,.stApp button,.stApp td,.stApp th{font-family:'Instrument Sans',sans-serif}
h1,h2,h3{font-family:'Fraunces',serif !important;font-weight:600 !important;letter-spacing:-.01em}
h1{font-size:2.6rem !important}
.lede{color:var(--muted);max-width:62ch;font-size:1.05rem;margin-bottom:.5rem}
.result{background:#fff;border:1px solid var(--line);border-left:6px solid var(--c);border-radius:10px;padding:1.4rem 1.6rem;margin-bottom:1rem}
.result .label{font-family:'Fraunces',serif;font-size:2.2rem;line-height:1.1;color:var(--c)}
.result .sub{color:var(--muted);margin-top:.35rem}
.meterwrap{position:relative;margin-top:1.4rem}
.meter{display:flex;height:14px;border-radius:7px;overflow:hidden}
.meter i{display:block;height:100%}
.pin{position:absolute;top:-5px;width:4px;height:24px;margin-left:-2px;border-radius:2px;background:var(--ink);box-shadow:0 0 0 2px #fff}
.zones{display:flex;margin-top:.4rem;font-size:.82rem;color:var(--muted)}
.zones span{text-align:center}
.empty{border:1.5px dashed var(--line);border-radius:10px;padding:2rem 1.5rem;color:var(--muted)}
.stFormSubmitButton button{width:100%}
"""


def html(s):
    st.markdown(" ".join(s.split()), unsafe_allow_html=True)


html(f"<style>{CSS}</style>")

MODEL_PATH = Path(__file__).parent / "model.joblib"
if not MODEL_PATH.exists():
    st.error("model.joblib not found. Put heart.csv in this folder and run `python train.py` first.")
    st.stop()


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


bundle = load()
model, best, res = bundle["model"], bundle["best"], bundle["results"]
data = bundle["data"]

LABELS = {"Age": "Age", "Sex": "Sex", "ChestPainType": "Chest pain type",
          "RestingBP": "Resting blood pressure", "Cholesterol": "Cholesterol",
          "FastingBS": "Fasting sugar > 120", "RestingECG": "Resting ECG",
          "MaxHR": "Max heart rate", "ExerciseAngina": "Exercise angina",
          "Oldpeak": "ST depression", "ST_Slope": "ST slope"}
CP = {"ASY": "Asymptomatic", "ATA": "Atypical angina", "NAP": "Non-anginal pain", "TA": "Typical angina"}
ECG = {"Normal": "Normal", "ST": "ST-T wave abnormality", "LVH": "Left ventricular hypertrophy"}
SLOPE = {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}

st.title("Heart disease risk screening")
html(f'<p class="lede">Enter a patient’s clinical measurements to estimate the probability of heart '
     f'disease. {best} scored best of three models trained on {len(data)} patients.</p>')

tab_assess, tab_perf, tab_data = st.tabs(["Assess a patient", "Model performance", "Explore the data"])

# ---------------------------------------------------------------- Assess
with tab_assess:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.form("patient"):
            st.subheader("Profile")
            a, b = st.columns(2)
            age = a.number_input("Age", 18, 100, 54)
            sex = b.selectbox("Sex", ["Female", "Male"], index=1)
            cp = st.selectbox("Chest pain type", list(CP), format_func=CP.get)

            st.subheader("Blood tests and vitals")
            a, b = st.columns(2)
            trestbps = a.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
            chol = b.number_input("Cholesterol (mg/dl)", 100, 600, 240)
            chol_unknown = st.checkbox("Cholesterol not measured")
            a, b = st.columns(2)
            fbs = a.selectbox("Fasting blood sugar above 120 mg/dl", ["No", "Yes"])
            restecg = b.selectbox("Resting ECG", list(ECG), format_func=ECG.get)

            st.subheader("Exercise test")
            a, b = st.columns(2)
            thalach = a.number_input("Max heart rate reached (bpm)", 60, 220, 150)
            exang = b.selectbox("Chest pain during exercise", ["No", "Yes"])
            a, b = st.columns(2)
            oldpeak = a.number_input("ST depression (Oldpeak)", -3.0, 7.0, 1.0, step=0.1)
            slope = b.selectbox("Slope of peak exercise ST", list(SLOPE), index=1, format_func=SLOPE.get)

            submitted = st.form_submit_button("Estimate risk", type="primary")

    with right:
        if not submitted:
            html('<div class="empty">Fill in the patient’s details and select <b>Estimate risk</b>. '
                 'The result appears here.</div>')
        else:
            row = pd.DataFrame([{
                "Age": age, "RestingBP": trestbps,
                "Cholesterol": np.nan if chol_unknown else chol,
                "MaxHR": thalach, "Oldpeak": oldpeak,
                "Sex": "M" if sex == "Male" else "F", "ChestPainType": cp,
                "RestingECG": restecg, "ExerciseAngina": "Y" if exang == "Yes" else "N",
                "ST_Slope": slope, "FastingBS": int(fbs == "Yes")}])[bundle["features"]]
            p = float(model.predict_proba(row)[0, 1])
            band, color = (("Lower risk", LO) if p < 0.35 else
                           ("Moderate risk", MID) if p <= 0.65 else ("Higher risk", HI))
            html(f"""<div class="result" style="--c:{color}">
                <div class="label">{band}</div>
                <div class="sub">The model estimates a {p:.0%} probability of heart disease.</div>
                <div class="meterwrap">
                  <div class="meter"><i style="width:35%;background:{LO}"></i>
                  <i style="width:30%;background:{MID}"></i><i style="width:35%;background:{HI}"></i></div>
                  <b class="pin" style="left:{p * 100:.1f}%"></b></div>
                <div class="zones"><span style="width:35%">Lower</span>
                  <span style="width:30%">Moderate</span><span style="width:35%">Higher</span></div>
                </div>""")
            st.subheader("How this patient compares")
            med = bundle["medians"]
            cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
            st.table(pd.DataFrame({
                "This patient": [row[c].iloc[0] for c in cols],
                "Median, no disease": [med.loc[0, c] for c in cols],
                "Median, with disease": [med.loc[1, c] for c in cols],
            }, index=[LABELS[c] for c in cols]).astype(float).round(1))
            st.caption("Educational project. This is not a medical device and not a diagnosis.")

# ----------------------------------------------------------- Performance
with tab_perf:
    m = res.loc[best]
    st.subheader(f"{best} performed best")
    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['Accuracy']:.1%}")
    c[1].metric("Precision", f"{m['Precision']:.1%}")
    c[2].metric("Recall", f"{m['Recall']:.1%}")
    c[3].metric("AUC", f"{m['Test AUC']:.3f}")
    st.caption("Measured on the 20% of patients held out from training. "
               "The winner is chosen by 5-fold cross-validated AUC.")

    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#DCEBE6"))

    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.subheader("What the model relies on")
        imp = bundle["importance"].rename(index=LABELS).rename("Importance").reset_index()
        imp.columns = ["Feature", "Importance"]
        st.altair_chart(alt.Chart(imp).mark_bar(color=HI).encode(
            x=alt.X("Importance:Q", title="Drop in AUC when the feature is shuffled"),
            y=alt.Y("Feature:N", sort="-x", title=None)).properties(height=380))
    with b:
        st.subheader("Test-set predictions")
        st.table(pd.DataFrame(bundle["cm"],
                              index=["Actually no disease", "Actually disease"],
                              columns=["Predicted no disease", "Predicted disease"]))

# ------------------------------------------------------------------ Data
with tab_data:
    c = st.columns(3)
    c[0].metric("Patients", len(data))
    c[1].metric("With heart disease", f"{data['target'].mean():.0%}")
    c[2].metric("Features", len(bundle["features"]))

    feat = st.selectbox("Compare a feature by diagnosis", bundle["features"], format_func=LABELS.get)
    d = data.assign(Diagnosis=data["target"].map({0: "No disease", 1: "Disease"}))
    color = alt.Color("Diagnosis:N", scale=alt.Scale(domain=["No disease", "Disease"], range=[LO, HI]))
    if pd.api.types.is_numeric_dtype(d[feat]) and d[feat].nunique() > 6:
        chart = alt.Chart(d).mark_bar(opacity=0.75).encode(
            alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=16), title=LABELS[feat]),
            alt.Y("count()", stack=None, title="Patients"), color)
    else:
        chart = alt.Chart(d).mark_bar().encode(
            alt.X(f"{feat}:N", title=LABELS[feat]), alt.Y("count()", title="Patients"),
            alt.XOffset("Diagnosis:N"), color)
    st.altair_chart(chart.properties(height=340))
    st.caption("Cholesterol and blood pressure values recorded as 0 in the raw file are treated as missing.")
    with st.expander("View the cleaned data"):
        st.dataframe(data)