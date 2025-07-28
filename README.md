# 🌿 AI-Hydroponics: Fenugreek Harvest Predictor

An intelligent machine learning application that predicts optimal harvest timing for fenugreek plants in hydroponic systems. This project combines agricultural science with AI to help farmers maximize crop yield and timing.

## ✨ Live Demo

🚀 **[Try the Live Demo](your-streamlit-url-here)** - Interactive web application hosted on Streamlit Community Cloud

## 📊 Features

- **🤖 AI-Powered Predictions**: Random Forest ML model trained on biologically accurate growth data
- **📈 Real-time Analysis**: Interactive sliders for plant age and pH level input
- **📅 Harvest Forecasting**: Precise harvest date predictions with Indian timezone support
- **🎯 pH Optimization**: Visual charts showing optimal pH ranges for maximum growth
- **📱 Mobile-Friendly**: Responsive design works on all devices
- **⚡ Fast Performance**: Cached model loading for instant predictions

## 🧬 Model Details

### Machine Learning Approach
- **Algorithm**: Random Forest Regressor
- **Features**: Plant Age (days), pH Level (5.0-8.0)
- **Training Data**: 3,100+ biologically accurate data points
- **Optimization**: Parabolic pH sensitivity curve with optimal range 6.0-6.5

### Biological Accuracy
- **Growth Modeling**: Based on real fenugreek growth patterns
- **pH Sensitivity**: Scientifically validated optimal pH ranges
- **Maturity Timeline**: 30-42 day lifecycle modeling
- **Environmental Factors**: Age-dependent growth rate calculations

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)
1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and set main file to `app.py`
5. Click "Deploy" - Your app goes live in 2-3 minutes!

### Option 2: Local Development
```bash
# Clone the repository
git clone https://github.com/krishnabalajiwork/Ai-Hydroponics.git
cd Ai-Hydroponics

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Option 3: Hugging Face Spaces
1. Create account at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Streamlit Space
3. Upload `app.py` and `requirements.txt`
4. Your demo goes live automatically

## 📁 Project Structure

```
Ai-Hydroponics/
├── app.py                          # Main Streamlit web application
├── requirements.txt                 # Python dependencies
├── RandomForestRegressor.ipynb      # Original Jupyter notebook
├── README.md                        # Project documentation
└── outputs/                         # Sample output images (if any)
```

## 💻 Usage

### Web Application
1. **Access the live demo** using the Streamlit URL
2. **Adjust parameters** using the sidebar sliders:
   - Plant Age: 1-42 days
   - pH Level: 5.0-8.0
3. **View predictions** including:
   - Days until harvest
   - Expected harvest date
   - Growth rate calculations
   - pH condition assessment
4. **Analyze visualizations** showing pH optimization curves

### Jupyter Notebook
1. Open `RandomForestRegressor.ipynb` in Google Colab
2. Mount Google Drive and run all cells
3. Input plant age and pH when prompted
4. View console-based predictions

## 🎯 Use Cases

### For Hydroponic Farmers
- **Harvest Planning**: Optimize crop rotation and market timing
- **pH Management**: Maintain optimal growing conditions
- **Yield Prediction**: Plan resources and labor allocation
- **Quality Control**: Monitor plant health through growth rates

### For Agricultural Students
- **Learning Tool**: Understand pH impact on plant growth
- **Research Platform**: Analyze hydroponic growing patterns
- **Data Science**: Study ML applications in agriculture
- **Sustainability**: Explore efficient farming methods

## 🔬 Scientific Background

### Fenugreek Growing Requirements
- **Optimal pH Range**: 6.0-6.5 for maximum nutrient uptake
- **Growth Timeline**: 35-42 days from seed to harvest
- **Temperature**: 20-25°C ideal growing conditions
- **Nutrients**: Balanced NPK with micronutrient support

### Model Training Methodology
- **Data Generation**: Biologically validated growth curves
- **Feature Engineering**: pH penalty functions and age factors
- **Validation**: Cross-referenced with agricultural research
- **Performance**: Optimized for practical farming applications

## 🛠️ Technical Stack

- **Backend**: Python 3.8+
- **ML Framework**: scikit-learn RandomForestRegressor
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git/GitHub

## 📈 Performance Metrics

- **Model Accuracy**: High correlation with biological growth patterns
- **Prediction Speed**: <100ms response time
- **Data Coverage**: pH range 5.0-8.0, age range 1-42 days
- **Reliability**: Consistent results across parameter ranges

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:
- Additional plant varieties
- Advanced environmental factors (temperature, humidity)
- Historical data integration
- Mobile app development

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Chintha Krishna Balaji**
- BTech CSE Student
- GitHub: [@krishnabalajiwork](https://github.com/krishnabalajiwork)

## 🔮 Future Enhancements

- **Multi-crop Support**: Extend to other hydroponic crops
- **IoT Integration**: Real-time sensor data input
- **Advanced Analytics**: Historical growth tracking
- **Mobile App**: Native iOS/Android applications
- **API Development**: RESTful API for third-party integration

---

⭐ **Star this repository** if you find it helpful for your hydroponic farming or agricultural studies!