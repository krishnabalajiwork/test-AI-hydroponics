# 🌿 AI-Hydroponics — Fenugreek Harvest Predictor

![AI-Hydroponics Banner](https://img.shields.io/badge/AI--Hydroponics-Fenugreek%20Harvest%20Predictor-green?style=for-the-badge&logo=leaf)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Available-brightgreen?style=for-the-badge&logo=streamlit)](https://test-ai-hydroponics.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green?style=for-the-badge)](https://scikit-learn.org)

## 🌱 Project Overview

**AI-Hydroponics Fenugreek Harvest Predictor** is an intelligent agricultural application that helps hydroponic farmers optimize their fenugreek harvest timing using machine learning. By analyzing plant age and pH conditions, it provides precise predictions for optimal harvest dates, growth rates, and cultivation recommendations.

### 🎯 Problem Statement
Hydroponic farmers struggle to determine the optimal harvest timing for fenugreek crops, leading to:
* Reduced yield quality and quantity
* Inefficient resource utilization
* Inconsistent harvest planning
* pH imbalance affecting plant growth

### 💡 Solution
An AI-powered web application that:
* Predicts harvest timing with high accuracy based on data modeling
* Optimizes pH conditions for maximum growth
* Provides real-time growth rate calculations
* Offers science-based agricultural recommendations

---

## 💧 NFT Hydroponic System & Crop Parameters (Fenugreek / *Trigonella foenum-graecum*)

The dataset and predictive simulation model are built around real-world agronomic parameters specific to **Nutrient Film Technique (NFT)** hydroponics:

* **System Configuration:** Continuous thin-film nutrient solution flow over plant roots in sloped channels, ensuring high oxygenation and direct access to macro/micronutrients.
* **Optimal pH Range:** Maintained tightly between **6.0 and 6.5** to maximize the bioavailability of essential elements (phosphorus, iron, and micronutrients) while preventing nutrient lockout.
* **Lifecycle Duration:** A rapid 30-to-42-day seed-to-harvest window, highly sensitive to fluctuations in solution chemistry and ambient microclimate.
* **Quadratic pH Modeling:** The predictive backend models plant growth efficiency using a parabolic penalty function:
  $$\text{Efficiency} = 1.0 - 0.15 \times \vert{}\text{pH} - 6.2\vert{}$$
  Deviations outside the optimal window apply a controlled growth penalty, mirroring biological stress observed in real NFT setups.

---

## 🚀 Live Demo

🌐 **Try it now**: [https://test-ai-hydroponics.streamlit.app/](https://test-ai-hydroponics.streamlit.app/)

### Key Features:
* 🔮 **Real-time ML Predictions** — Instant harvest forecasting
* 📊 **Interactive pH Optimization** — Visual pH vs growth rate analysis
* 📅 **Growth Timeline** — Track plant development stages
* 🌿 **Plant-themed UI** — Beautiful, intuitive interface
* 📱 **Mobile Responsive** — Works smoothly across devices

---

## 🏗️ Technical Architecture

### Machine Learning Pipeline:
```text
Input Data ──> Feature Engineering ──> Random Forest Model ──> Prediction Output
    │                   │                       │                      │
    ▼                   ▼                       ▼                      ▼
Plant Age       pH Optimization         3,100+ Training        Harvest Date
pH Level        Curves                  Data Points            Growth Rate

```

### Technology Stack:

* **Frontend:** Streamlit (Interactive Web UI)
* **Backend:** Python 3.8+
* **Machine Learning:** Scikit-learn (Random Forest Regressor)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly, Matplotlib
* **Deployment:** Streamlit Community Cloud

---

## 📊 Model Performance & Biological Foundation

* **Algorithm:** Random Forest Regressor ($100$ estimators)
* **Training Corpus:** $3,100+$ biologically accurate synthetic and empirical data points
* **Input Features:** Plant age ($1$–$42$ days), pH level ($5.0$–$8.0$)
* **Base Maturity:** $35$ days under optimal conditions
* **Accuracy Profile:** Finely tuned for the biological optimal window of pH $6.0$–$6.5$

---

## 🛠️ Installation & Local Setup

### Prerequisites

* Python v3.8 or higher
* pip package manager

### 1. Clone & Install Dependencies

```bash
git clone [https://github.com/krishnabalajiwork/test-AI.git](https://github.com/krishnabalajiwork/test-AI.git)
cd test-AI
pip install -r requirements.txt

```

### 2. Run the Application

```bash
streamlit run app.py

```

Open your browser at `http://localhost:8501`.

---

## 📈 Usage Guide

1. **Input Parameters:** Use the sidebar sliders to select plant age ($1$–$42$ days) and configure the hydroponic solution's pH level ($5.0$–$8.0$).
2. **Generate Predictions:** Click **Predict Harvest** to instantly evaluate growth rate, days remaining, and estimated calendar harvest dates.
3. **Condition Assessment:** Review color-coded status badges (Green for optimal, Yellow/Orange for suboptimal, Red for poor conditions requiring intervention).

---

## 👨‍💻 Author & Contact

**Chintha Krishna Balaji**
<img width="485" height="608" alt="1752427516648" src="https://github.com/user-attachments/assets/0e37141c-ea53-4443-9857-c6934efd99b2" />
<img width="485" height="608" alt="1752427516648" src="https://github.com/user-attachments/assets/391cba6a-7c21-4cae-8bac-9c4593f43112" />


* **GitHub:** [@krishnabalajiwork](https://github.com/krishnabalajiwork)
* **Live Demo:** [test-ai-hydroponics.streamlit.app](https://test-ai-hydroponics.streamlit.app/)

---

## 📝 License

This project is open-source and released under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
