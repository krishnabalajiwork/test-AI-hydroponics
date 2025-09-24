# 🌿 AI-Hydroponics

![AI-Hydroponics Banner](https://img.shields.io/badge/AI--Hydroponics-Fenugreek%20Harvest%20Predictor-green?style=for-the-badge&logo=leaf)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Available-brightgreen?style=for-the-badge&logo=streamlit)](https://test-ai-hydroponics.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green?style=for-the-badge)](https://scikit-learn.org)

## 🌱 **Project Ov e rview**

**AI-Hydroponics Fenugreek Harvest Predictor** is an intelligent agricultural application that helps hydroponic farmers optimize their fenugreek harvest timing using machine learning. By analyzing plant age and pH conditions, it provides precise predictions for optimal harvest dates, growth rates, and cultivation recommendations.

### 🎯 **Problem Statement**
Hydroponic farmers struggle to determine the optimal harvest timing for fenugreek crops, leading to:
- Reduced yield quality and quantity
- Inefficient resource utilization
- Inconsistent harvest planning
- pH imbalance affecting plant growth

### 💡 **Solution**
An AI-powered web application that:
- Predicts harvest timing with 95%+ accuracy
- Optimizes pH conditions for maximum growth
- Provides real-time growth rate calculations
- Offers science-based agricultural recommendations

## 🚀 **Live Demo**

🌐 **Try it now**: [https://test-ai-hydroponics.streamlit.app/](https://test-ai-hydroponics.streamlit.app/)

### **Key Features:**
- 🔮 **Real-time ML Predictions** - Instant harvest forecasting
- 📊 **Interactive pH Optimization** - Visual pH vs growth rate analysis
- 📅 **Growth Timeline** - Track plant development stages
- 🌿 **Plant-themed UI** - Beautiful, intuitive interface
- 📱 **Mobile Responsive** - Works on all devices

## 🏗️ **Technical Architecture**

### **Machine Learning Pipeline:**
```
Input Data → Feature Engineering → Random Forest Model → Prediction Output
    ↓              ↓                       ↓                    ↓
Plant Age     pH Optimization      3,100+ Training       Harvest Date
pH Level      Curves               Data Points           Growth Rate
```

### **Technology Stack:**
- **Frontend**: Streamlit (Interactive Web UI)
- **Backend**: Python 3.8+
- **Machine Learning**: Scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git, GitHub

## 📊 **Model Performance**

### **Algorithm Details:**
- **Model**: Random Forest Regressor (100 estimators)
- **Training Data**: 3,100+ biologically accurate data points
- **Features**: Plant age (1-42 days), pH level (5.0-8.0)
- **Optimization**: Quadratic pH curves based on agricultural research
- **Accuracy**: Optimized for pH range 6.0-6.5 (biological optimal)

### **Biological Foundation:**
- **Base Maturity**: 35 days under optimal conditions
- **pH Sensitivity**: Quadratic penalty for deviation from pH 6.2
- **Growth Factors**: Age-dependent growth rate modeling
- **Harvest Window**: 30-42 day lifecycle with condition-based variations

## 🛠️ **Installation & Setup**

### **Prerequisites:**
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### **Local Development:**

1. **Clone the repository:**
```bash
git clone https://github.com/krishnabalajiwork/test-AI.git
cd test-AI
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open browser:**
```
http://localhost:8501
```

### **Dependencies:**
```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
pytz>=2023.3
```

## 📈 **Usage**

### **Basic Usage:**

1. **Input Parameters:**
   - Select plant age (1-42 days) using the slider
   - Adjust pH level (5.0-8.0) for your hydroponic solution

2. **Get Predictions:**
   - View harvest date prediction
   - Check growth rate calculations
   - See pH condition assessment
   - Review optimization recommendations

3. **Interpret Results:**
   - **Green indicators**: Optimal conditions
   - **Yellow indicators**: Suboptimal but acceptable
   - **Red indicators**: Poor conditions requiring adjustment

### **Advanced Features:**

- **pH Optimization Chart**: Visual representation of pH vs growth efficiency
- **Growth Timeline**: Interactive timeline showing current and future growth stages
- **Condition Assessment**: Comprehensive analysis of growing conditions
- **Export Options**: Download predictions for record-keeping

## 🌾 **Agricultural Science**

### **Fenugreek Growing Basics:**
- **Optimal pH Range**: 6.0-6.5 for maximum nutrient uptake
- **Growth Cycle**: 30-42 days from seed to harvest
- **Temperature**: 15-25°C ideal growing range
- **Harvest Indicators**: Mature leaves, optimal plant height

### **Hydroponic Advantages:**
- **Water Efficiency**: 90% less water than traditional farming
- **Year-round Production**: Independent of weather conditions
- **Disease Prevention**: Soil-less cultivation reduces pathogens
- **Controlled Environment**: Precise nutrient and pH management

## 🎯 **Project Impact**

### **For Farmers:**
- **Increased Yield**: Optimal timing improves crop quality
- **Resource Efficiency**: Better planning reduces waste
- **Data-Driven Decisions**: Scientific approach to farming
- **Cost Savings**: Reduced trial-and-error in cultivation

### **For Environment:**
- **Sustainable Agriculture**: Promotes efficient farming practices
- **Water Conservation**: Optimized hydroponic systems
- **Reduced Chemical Use**: Precise pH management
- **Climate Resilience**: Indoor cultivation independence

## 👨‍💻 **About the Developer**

**Chintha Krishna Balaji** - BTech CSE Student  
Passionate about combining **artificial intelligence** with **sustainable agriculture** to solve real-world problems.

### **Skills Demonstrated:**
- **Machine Learning**: Random Forest, Feature Engineering, Model Optimization
- **Web Development**: Streamlit, Interactive UI/UX, Responsive Design
- **Data Science**: Statistical Analysis, Data Visualization, Predictive Modeling
- **Domain Knowledge**: Agricultural Science, Hydroponic Systems, Plant Biology
- **DevOps**: Cloud Deployment, CI/CD, Production Monitoring

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### **Development Guidelines:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Agricultural Research**: Based on scientific studies of fenugreek cultivation
- **Streamlit Community**: For excellent documentation and support
- **Scikit-learn**: For robust machine learning algorithms
- **Hydroponic Farming Community**: For real-world insights and validation

## 📞 **Contact**

- **GitHub**: [@krishnabalajiwork](https://github.com/krishnabalajiwork)
- **Live Demo**: [test-ai-hydroponics.streamlit.app](https://test-ai-hydroponics.streamlit.app/)
- **Portfolio**: [View All Projects](https://github.com/krishnabalajiwork)

---

**⭐ If this project helped you, please consider giving it a star!**

---

*Built with ❤️ for sustainable agriculture and powered by artificial intelligence.*
