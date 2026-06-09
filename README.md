## Crop Yield Prediction in Indian States

This notebook presents an end-to-end machine learning project aimed at predicting crop yields in various Indian states based on environmental and agricultural factors. The project includes data loading, preprocessing, feature engineering, model training (RandomForestRegressor), evaluation, and a user-friendly Gradio interface for making predictions.

### 1. Project Overview

The primary goal of this project is to develop a predictive model for crop yield. Understanding the factors influencing crop yield is crucial for agricultural planning, resource allocation, and food security. The model considers various features like crop type, state, annual rainfall, fertilizer and pesticide usage, and different seasons to predict the expected yield.

### 2. Dataset

The dataset used for this project is sourced from Kaggle: "Crop Yield in Indian States Dataset". It contains **19689 entries** and includes information on:
- **Crop**: Type of crop (e.g., Rice, Wheat, Maize)
- **Crop_Year**: Year of cultivation
- **Season**: Cultivation season (e.g., Kharif, Rabi, Whole Year)
- **State**: Indian state where cultivation occurred
- **Area**: Cultivated area
- **Production**: Total production
- **Annual_Rainfall**: Annual rainfall in mm
- **Fertilizer**: Amount of fertilizer used
- **Pesticide**: Amount of pesticide used
- **Yield**: Crop yield (target variable)

### 3. Data Preprocessing and Feature Engineering

#### Categorical Feature Encoding
- **Ordinal Encoding**: 'Crop' and 'State' features were encoded using `OrdinalEncoder` due to their ordinal nature in relation to potential yield differences.
- **One-Hot Encoding**: 'Season' was One-Hot Encoded to represent different seasons without implying any ordinal relationship.

#### Numerical Feature Scaling
- Numerical features such as 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', and 'Pesticide' were scaled using `StandardScaler` to normalize their ranges and improve model performance.

#### Feature Engineering
New features were created to capture more complex relationships:
- **Fertilizer_Usage**: `Fertilizer / Area` (fertilizer per unit area)
- **Pesticide_Usage**: `Pesticide / Area` (pesticide per unit area)
- **Rain_Fert_Interaction**: `Annual_Rainfall * Fertilizer_Usage` (interaction between rainfall and fertilizer usage)

#### Handling Data Leakage
Initially, 'Production' and 'Area' were found to be highly correlated with 'Yield', leading to data leakage. These features were subsequently dropped from the dataset (`x=x.drop(['Production','Area'],axis=1)`) to build a more robust and realistic predictive model.

### 4. Model Training and Evaluation

#### Model Used
- A `RandomForestRegressor` was chosen for its robustness and ability to handle complex non-linear relationships. Hyperparameters were set as `max_depth=20`, `n_estimators=100`, and `random_state=42`.

#### Training Process
- The data was split into training and testing sets (`80%` train, `20%` test).
- The model was trained on the preprocessed training data.

#### Evaluation Metrics
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
- **R-squared (R2 Score)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

**Initial Model Performance (with data leakage):**
- R2 Score: `0.905`
- Mean Squared Error (MSE): `75727.795`
- Mean Absolute Error (MAE): `10.769`

After addressing data leakage by removing 'Production' and 'Area', the model's performance significantly improved:

**Final Model Performance (without data leakage):**
- R2 Score: `0.975`
- Mean Squared Error (MSE): `19835.208`
- Mean Absolute Error (MAE): `11.148`

### 5. Deployment and Interface (Gradio)

A user-friendly web interface was built using the Gradio library to allow interactive predictions. The interface takes the following inputs:
- Crop Type
- State
- Crop Year
- Annual Rainfall
- Fertilizer Amount
- Pesticide Amount
- Season (Checkbox for Kharif, Rabi, Summer, Whole Year, Winter)

The interface then outputs the predicted crop yield.

### 6. Files

- `yield_model.pkl`: The trained machine learning pipeline (including preprocessing steps and the Random Forest Regressor) is saved as a pickle file for easy deployment and inference.
