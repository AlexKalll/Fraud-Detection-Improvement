# Fraud Detection Improvement Project Report  

**Project Title:** Improved Detection of Fraud Cases for E-commerce and Bank Transactions  
**Date:** July 20, 2025 (Ongoing)  

---

## 1. Project Overview  
This project is undertaken for **Adey Innovations Inc.**, a leading financial technology company, with the core objective of significantly enhancing fraud detection capabilities for both e-commerce and bank credit card transactions. In an increasingly digital financial landscape, sophisticated fraudulent activities pose a continuous threat, leading to substantial financial losses, reputational damage, and erosion of customer trust.  

The initiative focuses on:  
- Performing comprehensive **data analysis and preprocessing**  
- Engineering advanced **features** (geolocation insights, time-based metrics)  
- Implementing techniques to handle **class imbalance**  
- Building and evaluating **machine learning models**  
- Utilizing **Explainable AI (XAI)** tools like SHAP  

**Key Challenge:**  
Balancing security (minimizing false negatives) and user experience (minimizing false positives).  

---

## 2. Data Acquisition and Understanding  
### Datasets Used:  
| Dataset | Path | Description |  
|---------|------|-------------|  
| E-commerce Transactions | `data/Fraud_Data.csv` | User, device, time, value, source, browser, IP address, and fraud label (`class`) |  
| Credit Card Transactions | `data/creditcard.csv` | Anonymized features (`V1-V28`), `Time`, `Amount`, and fraud label (`Class`) |  
| IP to Country Mapping | `data/IpAddress_to_Country.csv` | Maps IP ranges to countries |  

### Key EDA Insights:  
- **Class Imbalance**:  
  - E-commerce: 9.36% fraud  
  - Credit Card: 0.17% fraud  
- **Transaction Values**:  
  - Heavy right-skew for both datasets  
  - Fraudulent transactions often smaller (card testing behavior)  
- **Geolocation**:  
  - High fraud rates in specific countries (Turkmenistan, Namibia, etc.)  

---

## 3. Data Preprocessing  
### Steps Performed (`src/data_preprocessing.py`):  
1. **Data Cleaning**:  
   - Dropped missing values  
   - Removed duplicates (1,081 rows from credit card data)  
2. **Data Type Conversion**:  
   ```python
   # Example conversions:
   fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
   ip_data[['lower_bound_ip', 'upper_bound_ip']] = ip_data[['lower_bound_ip', 'upper_bound_ip']].astype('int64')
   ```  
3. **Geolocation Merging**:  
   - Used `pd.merge_asof()` for efficient IP-to-country mapping  

---

## 4. Feature Engineering (`src/feature_engineering.py`)  
### Engineered Features for E-commerce Data:  
| Feature Type | Examples | Rationale |  
|-------------|----------|-----------|  
| **Time-Based** | `hour_of_day`, `day_of_week`, `time_since_signup` | Fraudsters often operate at specific times or exploit new accounts |  
| **Behavioral** | `user_transaction_count`, `device_velocity` | High-frequency activity may indicate automated fraud |  

**Optimization**:  
- Replaced slow `groupby().apply()` with vectorized `pandas.rolling()` for velocity features  

---

## 5. Data Transformation  
### Categorical Encoding:  
```python
# One-Hot Encoding for categorical features
fraud_data_encoded = pd.get_dummies(
    fraud_data, 
    columns=['source', 'browser', 'sex', 'country'], 
    drop_first=True
)
```  
### Numerical Scaling:  
- Applied `StandardScaler` to all numerical features  

---

## 6. Challenges & Solutions  
| Challenge | Solution |  
|-----------|----------|  
| Class imbalance | Planned: SMOTE oversampling on training data only |  
| IP-to-country merge | Used `pd.merge_asof()` with sorted IP ranges |  
| Velocity feature speed | Implemented vectorized `rolling()` window |  

---

## 7. Next Steps  
1. **Model Development** (`03_Modeling.ipynb`):  
   - Train Logistic Regression + Ensemble (XGBoost/LightGBM)  
   - Evaluate using AUC-PR/F1-score  
2. **Explainability** (`04_Explainability.ipynb`):  
   - Generate SHAP plots for model interpretation  

---

## 8. Conclusion  
The project has successfully:  
✔ Cleaned and enriched datasets  
✔ Engineered predictive features  
✔ Optimized computational bottlenecks  
