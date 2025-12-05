# California Housing Price Prediction

A machine learning project that predicts median house values in California using Linear Regression with multiple improvement techniques and regularization methods.

## Project Overview

This project analyzes the California Housing dataset to build predictive models for house prices. It explores feature selection, model improvement techniques, and compares different regression approaches including standard Linear Regression, Ridge, and Lasso regression.


## Live Demo

**Streamlit App:** [https://housepricinglinearregression-bowrjtyqkniqtg95kb7scj.streamlit.app/](https://housepricinglinearregression-bowrjtyqkniqtg95kb7scj.streamlit.app/)

**Google Colab Notebook:** [https://colab.research.google.com/drive/1Afy5tFNvmCqEiza3XGs_Ndhcgh2isYuQ#scrollTo=690f55ce](https://colab.research.google.com/drive/1Afy5tFNvmCqEiza3XGs_Ndhcgh2isYuQ#scrollTo=690f55ce)

## Features

- **Feature Selection:** Identifies and uses the top 3 most correlated features (Median Income, Average Rooms, Latitude)
- **Model Improvement Techniques:**
  - Feature Scaling using StandardScaler
  - Polynomial Features (degree 2) for capturing non-linear relationships
- **Model Comparison:** Cross-validation comparison of Linear Regression, Ridge, and Lasso
- **Interactive Web App:** Streamlit dashboard for real-time predictions
- **Comprehensive Visualizations:** Correlation plots, residual analysis, and performance comparisons

<!-- Add comparison chart here -->
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/71aed1bd-afa2-431e-8e84-4ec54d5bcc35" />

## Project Structure

```
├── house_pricing.ipynb           
├── house_pricing_app.py       
├── requirements.txt             
├── README.md                     

```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd california-housing-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `house_pricing.ipynb` and run all cells

### Running the Streamlit App

1. Start the Streamlit server:
```bash
streamlit run house_pricing_app.py
```

2. The app will open in your browser at `http://localhost:8501`

3. Adjust the input sliders and click "Predict House Value" to get predictions


## Model Performance

| Model | Test RMSE | Test R² | Test MAE |
|-------|-----------|---------|----------|
| Baseline Linear Regression | 0.8332 | 0.4703 | 0.6215 |
| Linear Regression + Scaling | 0.8332 | 0.4703 | 0.6215 |
| **Linear Regression + Polynomial** | **0.8294** | **0.4750** | **0.6076** |
| Ridge Regression | 0.8332 | 0.4703 | 0.6215 |
| Lasso Regression | 0.8362 | 0.4664 | 0.6285 |

**Best Model:** Linear Regression with Polynomial Features (degree 2)

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/f9df3bb6-a41f-4e7a-82ec-dad9ae1d6a8f" />


## Key Findings

1. **No Overfitting:** All models show similar training and testing performance, indicating good generalization
2. **Feature Importance:** Median Income has the strongest correlation (0.688) with house prices
3. **Polynomial Features:** Adding polynomial terms improved performance slightly (RMSE: 0.8294)
4. **Regularization:** Ridge and Lasso provided minimal benefit with only 3 features
5. **Model Limitation:** R² of 0.47 indicates moderate predictive power, suggesting additional features could improve performance

## Technologies Used

- **Python 3.8+**
- **Scikit-learn:** Machine learning algorithms and dataset
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computing
- **Matplotlib & Seaborn:** Data visualization
- **Streamlit:** Web application framework
- **Jupyter Notebook:** Interactive development environment

## Dataset

The California Housing dataset contains information from the 1990 California census with 20,640 samples and 8 features:

- **MedInc:** Median income in block group
- **HouseAge:** Median house age in block group
- **AveRooms:** Average number of rooms per household
- **AveBedrms:** Average number of bedrooms per household
- **Population:** Block group population
- **AveOccup:** Average number of household members
- **Latitude:** Block group latitude
- **Longitude:** Block group longitude
- **Target:** Median house value (in $100,000s)

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/091d146a-3707-4876-89e7-47d03f0cbea0" />


## Acknowledgments

- Dataset provided by Scikit-learn library
- Original dataset from the 1990 U.S. Census
