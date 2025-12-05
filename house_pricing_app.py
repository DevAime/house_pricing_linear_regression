import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page configuration
st.set_page_config(page_title="California Housing Price Predictor", layout="wide")

# Title and description
st.title("California Housing Price Predictor")
st.markdown("""
This app predicts median house values in California using a Linear Regression model with Polynomial Features.
The model uses three key features: Median Income, Average Rooms, and Latitude.
""")

# Load and prepare data
@st.cache_data
def load_and_train_model():
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = pd.Series(california.target, name='MedHouseVal')
    
    # Select top 3 features
    top_3_features = ['MedInc', 'AveRooms', 'Latitude']
    X_top3 = X[top_3_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_top3, y, test_size=0.2, random_state=42
    )
    
    # Train polynomial model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Calculate metrics
    y_pred_test = model.predict(X_test_poly)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Get feature statistics for reference
    feature_stats = X_top3.describe()
    
    return model, poly, test_rmse, test_r2, test_mae, feature_stats, X_test, y_test, y_pred_test

# Load model and data
model, poly, test_rmse, test_r2, test_mae, feature_stats, X_test, y_test, y_pred_test = load_and_train_model()

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Features")
    st.markdown("Enter the property details below:")
    
    # Input fields with helpful descriptions
    med_inc = st.slider(
        "Median Income (in $10,000s)",
        min_value=0.5,
        max_value=15.0,
        value=3.5,
        step=0.1,
        help="Median income of households in the block group"
    )
    
    ave_rooms = st.slider(
        "Average Rooms per Household",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Average number of rooms per household"
    )
    
    latitude = st.slider(
        "Latitude",
        min_value=32.5,
        max_value=42.0,
        value=37.0,
        step=0.1,
        help="Latitude coordinate of the property location"
    )
    
    # Predict button
    if st.button("Predict House Value", type="primary"):
        # Prepare input
        input_data = np.array([[med_inc, ave_rooms, latitude]])
        input_poly = poly.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_poly)[0]
        
        # Display prediction
        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric(
            label="Predicted Median House Value",
            value=f"${prediction * 100000:,.0f}",
            help="Predicted value in USD (model output is in $100,000s)"
        )
        
        st.info(f"The model predicts this property is worth approximately **${prediction * 100000:,.0f}**")

with col2:
    st.header("Model Performance")
    
    # Display metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("RMSE", f"{test_rmse:.4f}")
    
    with metric_col2:
        st.metric("R² Score", f"{test_r2:.4f}")
    
    with metric_col3:
        st.metric("MAE", f"{test_mae:.4f}")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2 = st.tabs(["Prediction vs Actual", "Feature Statistics"])
    
    with tab1:
        st.subheader("Model Predictions vs Actual Values")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred_test, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Polynomial Model: Predicted vs Actual')
        st.pyplot(fig)
        
        st.caption("Points closer to the red line indicate better predictions")
    
    with tab2:
        st.subheader("Feature Statistics from Dataset")
        st.dataframe(feature_stats, use_container_width=True)
        
        st.caption("Use these statistics as reference when selecting input values")

# Additional information section
st.markdown("---")
st.header("About the Model")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.subheader("Model Details")
    st.markdown("""
    - **Model Type:** Linear Regression with Polynomial Features (degree 2)
    - **Features Used:** Median Income, Average Rooms, Latitude
    - **Training Data:** California Housing Dataset (20,640 samples)
    - **Test Set Performance:** RMSE = 0.8294, R² = 0.4750
    """)

with info_col2:
    st.subheader("How to Use")
    st.markdown("""
    1. Adjust the sliders to input property characteristics
    2. Click "Predict House Value" to get the prediction
    3. The model outputs values in hundreds of thousands of dollars
    4. Review the model performance metrics and visualizations
    """)

st.markdown("---")
st.caption("Data source: California Housing Dataset from Scikit-Learn")