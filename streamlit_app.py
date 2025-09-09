#!/usr/bin/env python3
"""
Streamlit Web Application for Wastewater Quality Prediction
This application provides an interactive interface for predicting outlet water quality parameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from train_model import WastewaterQualityModel
from data_preprocessing import get_sample_data, prepare_model_data

# Configure Streamlit page
st.set_page_config(
    page_title="Wastewater Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        if os.path.exists('wastewater_model.pkl'):
            return WastewaterQualityModel.load_model('wastewater_model.pkl')
        else:
            st.error("Model file not found. Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for visualization."""
    data = get_sample_data()
    X, y, feature_cols, target_cols = prepare_model_data(data)
    return data, X, y, feature_cols, target_cols

def main():
    """Main Streamlit application."""
    
    # Title and header
    st.title("💧 Wastewater Quality Prediction System")
    st.markdown("""
    This application predicts outlet water quality parameters based on inlet conditions and weather data.
    Enter the parameters in the sidebar and click **Predict** to get the results.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar for input parameters
    st.sidebar.header("🔧 Input Parameters")
    st.sidebar.markdown("Enter the inlet water quality parameters and weather conditions:")
    
    # Create input fields in sidebar
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.subheader("Water Quality")
        bod = st.number_input("BOD (mg/l)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)
        cod = st.number_input("COD (mg/l)", min_value=0.0, max_value=1000.0, value=450.0, step=1.0)
        tds = st.number_input("TDS (mg/l)", min_value=0.0, max_value=3000.0, value=1400.0, step=10.0)
        ec = st.number_input("EC (mS/cm)", min_value=0.0, max_value=10.0, value=2.1, step=0.01)
    
    with col2:
        st.subheader("Chemical Parameters")
        nh4 = st.number_input("NH4 (mg/l)", min_value=0.0, max_value=50.0, value=9.0, step=0.1)
        no3 = st.number_input("NO3 (mg/l)", min_value=0.0, max_value=100.0, value=21.0, step=0.1)
        do = st.number_input("DO", min_value=0.0, max_value=20.0, value=0.5, step=0.1)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=8.7, step=0.1)
    
    st.sidebar.subheader("Weather Conditions")
    tavg = st.sidebar.number_input("Temperature Avg (°C)", min_value=-10.0, max_value=50.0, value=28.0, step=0.1)
    
    # Prediction button
    predict_button = st.sidebar.button("🔮 Predict Outlet Quality", type="primary")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Results", "📈 Feature Importance", "🔍 Model Performance", "📁 Batch Prediction"])
    
    with tab1:
        st.header("Prediction Results")
        
        if predict_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'BOD_inlet': [bod],
                'COD_inlet': [cod],
                'TDS_inlet': [tds],
                'EC_inlet': [ec],
                'NH4_inlet': [nh4],
                'NO3_inlet': [no3],
                'DO_inlet': [do],
                'pH_inlet': [ph],
                'tavg': [tavg]
            })
            
            try:
                # Make prediction
                predictions = model.predict(input_data)
                
                # Display results in a nice table
                st.success("✅ Prediction completed successfully!")
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Parameter': ['BOD (mg/l)', 'COD (mg/l)', 'TDS (mg/l)', 'EC (mS/cm)', 
                                'NH4 (mg/l)', 'NO3 (mg/l)', 'DO', 'pH'],
                    'Inlet Value': [bod, cod, tds, ec, nh4, no3, do, ph],
                    'Predicted Outlet': [
                        f"{predictions['BOD_outlet'].iloc[0]:.2f}",
                        f"{predictions['COD_outlet'].iloc[0]:.2f}",
                        f"{predictions['TDS_outlet'].iloc[0]:.2f}",
                        f"{predictions['EC_outlet'].iloc[0]:.2f}",
                        f"{predictions['NH4_outlet'].iloc[0]:.2f}",
                        f"{predictions['NO3_outlet'].iloc[0]:.2f}",
                        f"{predictions['DO_outlet'].iloc[0]:.2f}",
                        f"{predictions['pH_outlet'].iloc[0]:.2f}"
                    ]
                })
                
                # Calculate removal efficiency for relevant parameters
                removal_efficiency = []
                for i, param in enumerate(['BOD', 'COD', 'TDS', 'EC', 'NH4', 'NO3', 'DO', 'pH']):
                    inlet_val = results_df['Inlet Value'].iloc[i]
                    outlet_val = float(results_df['Predicted Outlet'].iloc[i])
                    
                    if param in ['BOD', 'COD', 'TDS', 'EC', 'NH4', 'NO3']:
                        # For these parameters, higher removal is better
                        efficiency = ((inlet_val - outlet_val) / inlet_val) * 100
                        removal_efficiency.append(f"{efficiency:.1f}%")
                    elif param == 'DO':
                        # For DO, increase is better
                        increase = ((outlet_val - inlet_val) / inlet_val) * 100
                        removal_efficiency.append(f"+{increase:.1f}%")
                    else:
                        # For pH, show the change
                        change = outlet_val - inlet_val
                        removal_efficiency.append(f"{change:+.2f}")
                
                results_df['Treatment Efficiency'] = removal_efficiency
                
                # Display the table
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Create visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart comparison
                    fig = go.Figure(data=[
                        go.Bar(name='Inlet', x=results_df['Parameter'], y=results_df['Inlet Value']),
                        go.Bar(name='Predicted Outlet', x=results_df['Parameter'], 
                              y=[float(x) for x in results_df['Predicted Outlet']])
                    ])
                    fig.update_layout(
                        title='Inlet vs Predicted Outlet Values',
                        xaxis_title='Parameters',
                        yaxis_title='Values',
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Treatment efficiency chart
                    efficiency_values = []
                    for eff in removal_efficiency:
                        if '%' in eff:
                            efficiency_values.append(float(eff.replace('%', '').replace('+', '')))
                        else:
                            efficiency_values.append(float(eff) * 10)  # Scale pH changes
                    
                    fig2 = go.Figure(data=[
                        go.Bar(x=results_df['Parameter'], y=efficiency_values,
                              marker_color=['green' if x > 0 else 'red' for x in efficiency_values])
                    ])
                    fig2.update_layout(
                        title='Treatment Efficiency',
                        xaxis_title='Parameters',
                        yaxis_title='Efficiency (%)',
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.info("👈 Enter parameters in the sidebar and click 'Predict' to see results.")
    
    with tab2:
        st.header("Feature Importance Analysis")
        
        if model and model.feature_importance:
            # Create feature importance plot
            features = list(model.feature_importance.keys())
            importances = list(model.feature_importance.values())
            
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(x=importances, y=features, orientation='h',
                      marker_color='skyblue')
            ])
            fig.update_layout(
                title='Feature Importance for Wastewater Quality Prediction',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=500,
                margin=dict(l=100)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Feature Importance Interpretation:**
            - Higher values indicate more important features for prediction
            - pH and TDS are typically the most influential parameters
            - Weather conditions (tavg) also play a significant role
            """)
        else:
            st.warning("Feature importance data not available.")
    
    with tab3:
        st.header("Model Performance Metrics")
        
        # Load sample data for visualization
        try:
            data, X, y, feature_cols, target_cols = load_sample_data()
            
            # Display model statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Type", "Random Forest Regressor")
                st.metric("Number of Features", len(feature_cols))
                st.metric("Number of Targets", len(target_cols))
                st.metric("Training Data Points", len(data))
            
            with col2:
                # If we have model performance metrics, display them
                st.info("""
                **Model Performance:**
                - Random Forest with 100 estimators
                - Multi-output regression for 8 parameters
                - Feature scaling applied
                - Cross-validation used for robust training
                """)
            
            # Show correlation heatmap
            st.subheader("Parameter Correlation Matrix")
            correlation_matrix = data[feature_cols + target_cols].corr()
            
            fig = px.imshow(correlation_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale="RdBu_r",
                          title="Correlation Matrix of Water Quality Parameters")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading performance data: {str(e)}")
    
    with tab4:
        st.header("Batch Prediction")
        st.markdown("Upload an Excel file with multiple samples for batch prediction.")
        
        # File upload
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_excel(uploaded_file)
                
                st.subheader("Uploaded Data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check if required columns exist
                required_cols = ['BOD_inlet', 'COD_inlet', 'TDS_inlet', 'EC_inlet', 
                               'NH4_inlet', 'NO3_inlet', 'DO_inlet', 'pH_inlet', 'tavg']
                
                if all(col in df.columns for col in required_cols):
                    if st.button("🔄 Process Batch Prediction"):
                        # Make batch predictions
                        predictions = model.predict(df[required_cols])
                        
                        # Combine input and predictions
                        result_df = pd.concat([df[required_cols], predictions], axis=1)
                        
                        st.subheader("Batch Prediction Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download button
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="wastewater_predictions.csv",
                            mime="text/csv"
                        )
                else:
                    st.error(f"Missing required columns. Expected: {required_cols}")
                    st.error(f"Found columns: {list(df.columns)}")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Template download
        st.subheader("📋 Download Template")
        st.markdown("Download a template file to see the expected format:")
        
        template_data = {
            'BOD_inlet': [200, 190, 210],
            'COD_inlet': [450, 420, 480],
            'TDS_inlet': [1400, 1350, 1450],
            'EC_inlet': [2.1, 2.0, 2.2],
            'NH4_inlet': [9.0, 8.5, 9.5],
            'NO3_inlet': [21.0, 20.0, 22.0],
            'DO_inlet': [0.5, 0.4, 0.6],
            'pH_inlet': [8.7, 8.6, 8.8],
            'tavg': [28.0, 27.0, 29.0]
        }
        
        template_df = pd.DataFrame(template_data)
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv_template,
            file_name="wastewater_template.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🌊 Wastewater Quality Prediction System | Built with Streamlit & scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
