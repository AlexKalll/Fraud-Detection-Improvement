
# Data for Fraud Detection Improvement Project

This directory contains the datasets used for the Fraud Detection Improvement project. Each dataset plays a crucial role in developing robust models for identifying fraudulent e-commerce and bank transactions.

---

## Datasets:

### 1. `Fraud_Data.csv`

This dataset contains e-commerce transaction data, primarily used for identifying fraudulent activities in online purchases. 

**Columns:**
* **`user_id`**: A unique identifier for the user who made the transaction. 
* **`signup_time`**: The timestamp when the user signed up on the platform. 
* **`purchase_time`**: The timestamp when the purchase was made. 
* **`purchase_value`**: The value of the purchase in US dollars. 
* **`device_id`**: A unique identifier for the device used to make the transaction. 
* **`source`**: The origin through which the user came to the site (e.g., 'SEO', 'Ads', 'Direct').
* **`browser`**: The web browser used for the transaction (e.g., 'Chrome', 'Safari', 'Firefox'). 
* **`sex`**: The gender of the user ('M' for male, 'F' for female). 
* **`age`**: The age of the user. 
* **`ip_address`**: The IP address from which the transaction was made. 
* **`class`**: The target variable. `1` indicates a fraudulent transaction, and `0` indicates a non-fraudulent (legitimate) transaction. 

**Critical Challenge:** This dataset exhibits a significant class imbalance, with a much smaller number of fraudulent transactions compared to legitimate ones. This characteristic necessitates careful selection of evaluation metrics and modeling techniques. 

---

### 2. `creditcard.csv`

This dataset contains anonymized bank transaction data, specifically curated for credit card fraud detection analysis. It features principal components obtained with PCA to protect sensitive information. 

**Columns:**
* **`Time`**: The number of seconds elapsed between this transaction and the first transaction in the dataset. [cite: 162]
* **`V1` to `V28`**: Anonymized features, which are the result of a PCA transformation. Their exact nature is not disclosed due to privacy, but they represent underlying patterns in the data. 
* **`Amount`**: The transaction amount in US dollars. 
* **`Class`**: The target variable. `1` indicates a fraudulent transaction, and `0` indicates a non-fraudulent transaction.

**Critical Challenge:** Similar to the e-commerce data, this dataset is extremely imbalanced, a common characteristic in real-world fraud detection problems. 

---

### 3. `IpAddress_to_Country.csv`

This dataset provides a mapping of IP address ranges to their corresponding countries. It is used to enrich the `Fraud_Data.csv` by adding geographical information based on the transaction IP addresses. 

**Columns:**
* **`lower_bound_ip_address`**: The lower bound of an IP address range. 
* **`upper_bound_ip_address`**: The upper bound of an IP address range. 
* **`country`**: The country associated with the given IP address range.
