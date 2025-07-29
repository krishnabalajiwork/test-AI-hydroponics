import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objects as go

# ----------  Load external CSS  ----------
def load_css(path: str):
    """Load CSS from the specified file path."""
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {path}")

# Apply styling
load_css("styles/main.css")

# Page configuration
st.set_page_config(
    page_title="🌿 AI-Hydroponics Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------  Data Generation  ----------
def calculate_days_to_maturity(age: int, ph: float) -> float:
    """
    Calculate days to maturity based on biological research findings.
    - Optimal pH: 6.0-6.5 (fastest growth)
    - pH effects: Growth slows outside optimal range
    - Total lifecycle: 30-42 days depending on conditions
    """
    base_maturity = 35
    ph_penalty = 8 * (ph - 6.2) ** 2
    age_factor = max(0, (25 - age) * 0.2)
    total = base_maturity + ph_penalty + age_factor
    return max(0, min(total - age, 42 - age))

@st.cache_data
def create_training_data() -> pd.DataFrame:
    ages, phs, days = [], [], []
    for age in range(1, 41):
        for ph in np.arange(5.0, 8.1, 0.1):
            ages.append(age)
            phs.append(round(ph, 1))
            days.append(calculate_days_to_maturity(age, ph))
    return pd.DataFrame({
        "Plant Age (Days)": ages,
        "pH Level": phs,
        "Days to Maturity": days
    })

# ----------  Model Training  ----------
@st.cache_resource
def train_model():
    df = create_training_data()
    X = df[["Plant Age (Days)", "pH Level"]]
    y = df["Days to Maturity"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df

def predict_fenugreek_harvest(age: int, ph: float, model) -> dict | str:
    """
    Predict fenugreek harvest details.
    Returns dict on success, or error message string.
    """
    if age < 1:
        return "❌ Invalid input: Plant Age must be ≥ 1 day."
    if not (5.0 <= ph <= 8.0):
        return "⚠️ pH outside tested range (5.0-8.0). Accuracy may vary."
    if age >= 42:
        return f"🌿 Age {age}d: Beyond maturity. Ready to harvest."
    if age >= 35 and 6.0 <= ph <= 7.0:
        return f"🌿 Age {age}d with optimal pH: Ready to harvest."

    inp = pd.DataFrame({"Plant Age (Days)": [age], "pH Level": [ph]})
    pred_days = model.predict(inp)[0]
    days_to_harvest = round(max(0, min(pred_days, 42 - age)), 1)
    harvest_day = age + days_to_harvest

    ph_eff = 1.0 - 0.15 * abs(ph - 6.2)
    growth_rate = (20 * ph_eff) / (age + days_to_harvest)

    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    harvest_date = (now + timedelta(days=days_to_harvest)).strftime("%A, %B %d, %Y")

    if 6.0 <= ph <= 6.5:
        status, color = "Optimal", "🟢"
    elif 5.8 <= ph <= 7.0:
        status, color = "Good", "🟡"
    elif 5.5 <= ph <= 7.5:
        status, color = "Suboptimal", "🟠"
    else:
        status, color = "Poor", "🔴"

    return {
        "current_date": now.strftime("%A, %B %d, %Y, %I:%M %p %Z"),
        "ph_status": f"{color} {status}",
        "growth_rate": f"{growth_rate:.4f} cm/day",
        "days_to_harvest": f"{days_to_harvest} days",
        "harvest_day": f"Day {harvest_day:.1f}",
        "harvest_date": harvest_date,
        "ph_efficiency": ph_eff
    }

# ----------  Charts  ----------
def create_ph_optimization_chart():
    ph_range = np.arange(5.0, 8.1, 0.1)
    eff = [1.0 - 0.15 * abs(p - 6.2) for p in ph_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ph_range, y=eff, mode="lines+markers",
                             line=dict(color="#38ef7d", width=3)))
    fig.update_layout(title="🌿 pH Optimization for Fenugreek Growth",
                      xaxis_title="pH Level", yaxis_title="Growth Efficiency",
                      template="plotly_white", height=400)
    fig.add_vrect(x0=6.0, x1=6.5, fillcolor="green", opacity=0.2,
                  annotation_text="Optimal Zone", annotation_position="top left")
    return fig

def create_growth_timeline(age: int, days_to_harvest: float):
    stages = [
        (0, 0), (7, 2), (14, 8), (21, 15), (35, 25)
    ]
    height = np.interp(age, [d for d, _ in stages], [h for _, h in stages])
    fig = go.Figure()
    past = [(d, h) for d, h in stages if d <= age] + [(age, height)]
    fig.add_trace(go.Scatter(x=[d for d, _ in past], y=[h for _, h in past],
                             mode="lines+markers", line=dict(color="#4caf50", width=4)))
    fig.add_trace(go.Scatter(x=[age, age + days_to_harvest],
                             y=[height, 25],
                             mode="lines+markers",
                             line=dict(color="#81c784", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=[age], y=[height],
                             mode="markers",
                             marker=dict(size=15, color="red", symbol="diamond")))
    fig.update_layout(title="🌱 Fenugreek Growth Timeline",
                      xaxis_title="Days", yaxis_title="Plant Height (cm)",
                      template="plotly_white", height=400)
    return fig

# ----------  Main App  ----------
def main():
    st.title("🌿 AI-Powered Fenugreek Hydroponics Predictor")

    model, df = train_model()
    with st.sidebar:
        st.header("🌱 Plant Parameters")
        age = st.slider("Plant Age (Days)", 1, 42, 23, help="Current plant age")
        ph = st.slider("pH Level", 5.0, 8.0, 6.0, 0.1, help="Hydroponic solution pH")
        st.markdown("---")
        st.write(f"**Data points:** {len(df):,}")
        st.write("**Model:** Random Forest")
        st.write("**Optimal pH:** 6.0–6.5")

    if st.button("🔮 Predict Harvest"):
        result = predict_fenugreek_harvest(age, ph, model)
        if isinstance(result, str):
            st.warning(result)
        else:
            cols = st.columns(3)
            cols[0].metric("📅 Current Date", result["current_date"])
            cols[1].metric("🌱 pH Condition", result["ph_status"])
            cols[2].metric("✅ Growth Rate", result["growth_rate"])
            cols2 = st.columns(3)
            cols2[0].metric("⏳ Days to Harvest", result["days_to_harvest"])
            cols2[1].metric("📅 Harvest Day", result["harvest_day"])
            cols2[2].metric("📆 Harvest Date", result["harvest_date"])
            st.plotly_chart(create_ph_optimization_chart(), use_container_width=True)
            st.plotly_chart(create_growth_timeline(age, float(result["days_to_harvest"].split()[0])),
                             use_container_width=True)

    with st.expander("🧠 About the AI Model"):
        st.write("""
        This predictor uses a Random Forest Regressor trained on synthetic, biologically-inspired data.
        It demonstrates the full ML pipeline from data generation through model deployment.
        """)
    with st.expander("🌿 About Fenugreek Hydroponics"):
        st.write("""
        Fenugreek is ideal for hydroponics: high yield, nutrient-rich, with minimal water usage.
        Maintain pH 6.0–6.5, provide 12–14h light, and monitor EC at 1.2–1.8 mS/cm.
        """)

if __name__ == "__main__":
    main()
