# 💧 Wastewater Quality Prediction System

A comprehensive machine learning web application for predicting outlet water quality parameters based on inlet conditions and weather data.

## 📋 Features

### Core Functionality
- **Real-time Prediction**: Predict outlet water quality parameters using inlet conditions and weather data
- **Interactive UI**: Clean, modern Streamlit interface with sidebar input controls
- **Multi-parameter Prediction**: Simultaneously predicts 8 outlet parameters:
  - BOD (mg/l)
  - COD (mg/l) 
  - TDS (mg/l)
  - EC (mS/cm)
  - NH4 (mg/l)
  - NO3 (mg/l)
  - DO
  - pH

### Advanced Features
- **Feature Importance Analysis**: Visual representation of which parameters most influence predictions
- **Treatment Efficiency Calculation**: Shows removal/improvement percentages for each parameter
- **Batch Prediction**: Upload Excel files for processing multiple samples at once
- **Model Performance Metrics**: Displays correlation matrices and model statistics
- **Template Download**: Provides properly formatted templates for batch uploads

## 🚀 Quick Start

### Installation

1. **Clone or download the project files**
2. **Create a virtual environment:**
   ```bash
   python3 -m venv wastewater_env
   source wastewater_env/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install pandas openpyxl streamlit scikit-learn xgboost joblib matplotlib seaborn plotly
   ```

### Running the Application

1. **Train the model (first time only):**
   ```bash
   python train_model.py
   ```

2. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** and navigate to the displayed URL (typically `http://localhost:8501`)

## 📊 Usage Instructions

### Single Prediction
1. **Enter Parameters**: Use the sidebar to input water quality and weather parameters
2. **Click Predict**: Press the "🔮 Predict Outlet Quality" button
3. **View Results**: Check the results table and visualizations

### Batch Prediction
1. **Go to Batch Prediction tab**
2. **Download template** to see the required format
3. **Upload your Excel file** with multiple samples
4. **Process and download results** as CSV

## 🧠 Model Details

- **Algorithm**: Random Forest Regressor with MultiOutput wrapper
- **Features**: 9 input parameters (8 water quality + 1 weather)
- **Targets**: 8 outlet water quality parameters
- **Performance**: Uses standardized features and cross-validation

## 📁 File Structure

```
wastewater_prediction/
├── streamlit_app.py           # Main Streamlit application
├── train_model.py            # Model training and evaluation
├── data_preprocessing.py     # Data loading and preprocessing
├── wastewater_model.pkl     # Trained model (generated)
└── README.md                # This documentation
```

---

**Built with ❤️ using Streamlit, scikit-learn, and Python**
