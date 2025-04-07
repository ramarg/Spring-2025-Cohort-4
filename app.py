import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Superstore Profit Predictor", layout="wide")
st.title("📊 Superstore Profit Prediction Tool")


@st.cache_resource
def load_and_process_data():
    # Load the dataset
    df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\Project_PM\\Superstore.csv", encoding='ISO-8859-1')

    df_clean = df.copy()

    cols_to_drop = ['Row ID', 'Order ID', 'Customer ID', 'Customer Name',
                    'Product ID', 'Product Name', 'Country', 'Postal Code']

    df_clean.drop(columns=cols_to_drop, inplace=True)

    # Convert date columns to datetime
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])
    df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'])

    # Create new features
    df_clean['Order_Month'] = df_clean['Order Date'].dt.month
    df_clean['Shipping_Duration'] = (df_clean['Ship Date'] - df_clean['Order Date']).dt.days

    # Drop the original date columns
    df_clean.drop(columns=['Order Date', 'Ship Date'], inplace=True)

    # Get categorical values
    categorical_columns = {
        'Segment': df_clean['Segment'].unique().tolist(),
        'Ship Mode': df_clean['Ship Mode'].unique().tolist(),
        'Category': df_clean['Category'].unique().tolist(),
        'Sub-Category': df_clean['Sub-Category'].unique().tolist(),
        'Region': df_clean['Region'].unique().tolist(),
        'State': df_clean['State'].unique().tolist()
    }

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    # Split features and target variable
    X = df_encoded.drop(columns=['Profit'])
    y = df_encoded['Profit']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Find reference categories (those that were dropped during one-hot encoding)
    reference_categories = {}
    for col, values in categorical_columns.items():
        for val in values:
            col_name = f"{col}_{val}"
            col_name = col_name.replace(" ", "_").replace("-", "_")
            if col_name not in X.columns:
                reference_categories[col] = val
                break

    return X_train, X_test, y_train, y_test, X, categorical_columns, reference_categories, df_clean


@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    # Initialize models
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42, verbosity=0)

    # Dictionary to store results
    results = {}

    # Train and evaluate models
    models = {
        "Linear Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {"model": model, "R2": r2, "MAE": mae, "RMSE": rmse}

    return results


# Load data and train models
try:
    X_train, X_test, y_train, y_test, X, categorical_columns, reference_categories, df_clean = load_and_process_data()
    model_results = train_models(X_train, y_train, X_test, y_test)

    # Use the best model (XGBoost)
    best_model = model_results["XGBoost"]["model"]

    # Main content
    st.write("This app predicts profit for Superstore transactions based on various inputs.")

    # Create sidebar for inputs
    st.sidebar.title("📝 Input Transaction Details")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Basic Info", "Product Info", "Location"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            sales = st.number_input("Sales ($)", min_value=0.0, value=100.0, step=10.0)
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

        with col2:
            discount = st.slider("Discount", min_value=0.0, max_value=0.9, value=0.0, step=0.05)
            order_month = st.slider("Order Month", min_value=1, max_value=12, value=6)
            shipping_duration = st.slider("Shipping Duration (days)", min_value=0, max_value=10, value=3)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            segment = st.selectbox("Segment", options=categorical_columns['Segment'])
            ship_mode = st.selectbox("Ship Mode", options=categorical_columns['Ship Mode'])

        with col2:
            category = st.selectbox("Category", options=categorical_columns['Category'])
            sub_category = st.selectbox("Sub-Category", options=categorical_columns['Sub-Category'])

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("Region", options=categorical_columns['Region'])
            state = st.selectbox("State", options=categorical_columns['State'])

        with col2:
            city = st.text_input("City", value="New York City")

    # Prediction button
    st.markdown("---")
    predict_button = st.button("🔍 Predict Profit", type="primary")

    if predict_button:
        # Create input data for prediction
        model_input = pd.DataFrame(0, index=[0], columns=X.columns)

        # Fill numeric values
        model_input['Sales'] = sales
        model_input['Quantity'] = quantity
        model_input['Discount'] = discount
        model_input['Order_Month'] = order_month
        model_input['Shipping_Duration'] = shipping_duration


        # Set categorical columns
        # Clean column names to match one-hot encoding format
        def clean_col_name(col, val):
            return f"{col}_{val}".replace(" ", "_").replace("-", "_")


        # Helper function to set categorical values
        def set_categorical(feature, value, ref_val):
            if value != ref_val:
                col = clean_col_name(feature, value)
                if col in model_input.columns:
                    model_input[col] = 1


        # Set categorical values
        set_categorical('Segment', segment, reference_categories.get('Segment'))
        set_categorical('Ship Mode', ship_mode, reference_categories.get('Ship Mode'))
        set_categorical('Category', category, reference_categories.get('Category'))
        set_categorical('Sub-Category', sub_category, reference_categories.get('Sub-Category'))
        set_categorical('Region', region, reference_categories.get('Region'))
        set_categorical('State', state, reference_categories.get('State'))

        # City (approximation - in real app would need to handle better)
        city_col = clean_col_name('City', city)
        if city_col in model_input.columns:
            model_input[city_col] = 1

        # Make prediction
        prediction = best_model.predict(model_input)[0]
        margin = prediction / sales * 100 if sales > 0 else 0

        # Display results
        st.markdown("## 📈 Prediction Results")

        # Create columns for results
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Profit", f"${prediction:.2f}")

        with col2:
            st.metric("Profit Margin", f"{margin:.2f}%")

        # Add insights and warnings
        if prediction < 0:
            st.error("⚠️ Warning: This transaction is predicted to lose money!")
        elif margin < 10 and sales > 0:
            st.warning("⚠️ Note: The profit margin is below 10%.")
        else:
            st.success("✅ This transaction is predicted to be profitable!")

        # Show feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            st.markdown("### 🔍 Top Factors Influencing Prediction")

            # Get feature importance
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(5)

            # Display as a bar chart
            st.bar_chart(feature_importance.set_index('Feature'))

    # Display model performance metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")

    for name, results in model_results.items():
        with st.sidebar.expander(f"{name} Metrics"):
            st.write(f"R² Score: {results['R2']:.4f}")
            st.write(f"MAE: ${results['MAE']:.2f}")
            st.write(f"RMSE: ${results['RMSE']:.2f}")

except Exception as e:
    st.error(f"Error loading data or training models: {e}")
    st.info("Make sure 'Superstore.csv' file is in the same directory as this script.")
    st.markdown("""
    ### Quick Fix
    1. Make sure the CSV file is named exactly 'Superstore.csv'
    2. Place it in the same folder as this script
    3. Restart the app
    """)