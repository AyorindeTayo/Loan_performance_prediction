# Loan_performance
Loan_performance
![Imgur](https://imgur.com/6GJxSXG.png)
# Loan Default Prediction Model

## Project Overview

This project builds a machine learning model to predict loan performance based on historical data. The model aims to assess the likelihood of loan defaults and late payments for a set of 2,000 loans currently under Due Diligence (test data). The process involves data loading, cleaning, preprocessing, merging datasets, and training a model using the Random Forest Classifier.

---

## Table of Contents

1. [Data Sources](#data-sources)
2. [Data Preprocessing](#data-preprocessing)
   - [Checking Data Types](#checking-data-types)
   - [Data Cleaning](#data-cleaning)
   - [Handling Missing Values](#handling-missing-values)
3. [Merging Datasets](#merging-datasets)
4. [Feature Engineering](#feature-engineering)
5. [Modeling Process](#modeling-process)
   - [Data Normalization](#data-normalization)
   - [Feature selection for prediction](#Feature_selection)
   - [Random forest classfication](#Random_classification)
   - [Cross-Validation of the Random forest classification](#cross-validation)
   - [Model Training](#model-training)
   - [Saving the training model](#Trained_model)
6. [Using the best-performed Model Evaluation on Test_loan_data](#model-evaluation)
7. [Predictions on Test Data](#predictions-on-test-data)
8. [Conclusion](#conclusion)

---

## Data Sources

The project utilizes the following datasets:

- **`train_loan_data.csv`**: Contains historical loan data.
- **`train_payment_data.csv`**: Contains payment history for the loans.
- **`data_for_ml1.csv`**: Test data with 2,000 loans for predictions.

All datasets are loaded and processed using Pandas within Google Colab.

---

## Data Preprocessing

### Checking Data Types

The datasets were loaded, and data types were checked to differentiate between numerical, categorical, and string columns. Non-essential columns with string data were removed to facilitate model training.

### Data Cleaning

- **String Columns Removal**: String-based columns were removed from `train_loan_data_cleaned` and `train_payment_data_cleaned`, except for categorical columns needed for the model.
- **Missing Value Replacement**: Missing values were replaced with `0` where necessary, ensuring that the dataset was complete.

### Handling Missing Values

After data cleaning, the datasets were checked for any missing values and handled accordingly to ensure no data integrity issues.

---

## Merging Datasets

To create a unified dataset for training the machine learning model, the cleaned versions of `train_loan_data` and `train_payment_data` were merged on the `loan_id` column. This combined dataset was saved as `merged_data.csv` and used for the following steps.

---

## Feature Engineering

In the combined dataset, we removed unnecessary columns such as `loan_id`, `business_id`, and others that were not directly related to model prediction. This left us with only relevant numerical and categorical features for training.

---

## Modeling Process

### Data Normalization

Data normalization was applied to ensure all features had a consistent scale, improving the performance of machine learning algorithms. The reason for normalization is to standardize feature ranges, which helps the model converge faster and produce better predictions.

## Feature Selection for Prediction

For our model, the following features were selected for predicting loan performance:

- **Features (Input Variables)**:
  - `principal`: The principal amount of the loan.
  - `total_owing_at_issue`: The total amount owed at the time of issue.
  - `employee_count`: The number of employees associated with the loan.
  - `total_recovered_on_time`: The total amount recovered on time.
  - `total_recovered_15_dpd`: The total amount recovered within 15 days past due.
  - `cash_yield_15_dpd`: The cash yield for loans 15 days past due.
  - **Additional Features**: 
- The additional features were extracted from the `train_payment_data` dataset to enrich the model's predictive capabilities. Below are the features and their descriptions:

- **`payment_count`**: The number of payments made on the loan. This feature was derived by aggregating payment records associated with each loan, providing insights into borrower payment behavior.
- **`total_paid`**: The total amount paid on the loan up to the point of analysis. This was calculated by summing the payment amounts for each loan, giving a comprehensive view of the repayment status.

These features were incorporated into the unified dataset during the merging process of `train_payment_data` and `train_loan_data` on the common identifier `loan_id`. This ensures that all relevant payment information is aligned with the corresponding loan records, enhancing the model's ability to predict loan performance effectively.

   

- **Target Feature**:
  - `paid_late`: Indicates whether the loan was paid late (binary outcome).

## Label Encoding

Label Encoding is a method that converts each category into a unique integer. This technique is particularly suitable for ordinal categorical variables where the order matters. For instance, in our dataset:

- `paid_late` might be encoded as:
  - `No` -> `0`
  - `Yes` -> `1`


## Model Training and Evaluation for Loan Default Prediction

This section outlines the training and validation process of the machine learning model for predicting loan defaults.

## Data Splitting

The dataset was divided into training and validation sets using an 80-20 split. This means that 80% of the data was used for training the model, while 20% was reserved for validation. This method ensures that the model's performance is evaluated on unseen data, providing a more accurate assessment of its generalization ability.

## Model Training

A **Random Forest Classifier** was selected for training due to its effectiveness in handling classification tasks. The model was trained using the training dataset, enabling it to learn patterns and relationships between features.

## Model Evaluation

After training, the model was evaluated on both the training and validation sets using the following metrics:

- **Accuracy**: The proportion of correctly predicted instances compared to the total instances. A high accuracy indicates that the model is performing well on the given dataset.
  
- **Confusion Matrix**: This matrix provides a breakdown of the true positives, true negatives, false positives, and false negatives, offering insights into the model's performance on each class.

- **Classification Report**: This report includes precision, recall, and F1-score for each class, providing a comprehensive overview of the model's performance across different metrics.

### Example Evaluation Metrics

- **Training Accuracy**: 100%
- **Validation Accuracy**: 90.27%
- **Confusion Matrix**: 


### Model Training

After cross-validation, the model was trained on the full dataset using the following steps:

- **Split the Data**: Features (`X`) and the target variable (`y`) were separated.
- **Train/Test Split**: The data was split into 80% training and 20% validation sets.
- **Train the Model**: A Random Forest Classifier with 100 estimators was trained on the training set.

---


The model was evaluated using several metrics:

- **Accuracy**: The overall accuracy of the model in predicting whether a loan was paid late.
- **Confusion Matrix**: A confusion matrix was generated to evaluate true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Precision, recall, and F1-score were used to assess the model’s performance.
- **ROC Curve**: The Receiver Operating Characteristic (ROC) curve was plotted to evaluate the model's predictive ability using the AUC score.

- The confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. 
- It allows visualization of the performance of an algorithm.
- It displays the number of correct and incorrect predictions broken down by each class.
- The confusion matrix in the above code was created to evaluate the performance of the Random Forest Classifier on the validation set.

- **True Positives (TP)**: The model correctly predicted that a loan would be paid late (paid_late = 1).
- **True Negatives (TN)**: The model correctly predicted that a loan would not be paid late (paid_late = 0).
- **False Positives (FP)**: The model incorrectly predicted that a loan would be paid late when it actually was not (paid_late = 0). Also known as a Type I error.
- **False Negatives (FN)**: The model incorrectly predicted that a loan would not be paid late when it actually was (paid_late = 1). Also known as a Type II error.

### Breakdown of the Confusion Matrix:

- **True Negatives (TN)**: 4896  
  This indicates the number of instances where the model correctly predicted that the loan was **not paid late**. 

- **False Positives (FP)**: 0  
  This means there were no instances where the model incorrectly predicted a loan as **paid late** when it was actually **not paid late**.

- **False Negatives (FN)**: 14  
  This represents the number of loans that were actually **paid late** but were incorrectly predicted by the model as **not paid late**.

- **True Positives (TP)**: 111  
  This indicates the number of instances where the model correctly predicted that the loan was **paid late**.

### Interpretation

From the confusion matrix, we can draw the following conclusions:

- The model achieved a high **True Negative** count, indicating strong performance in predicting loans that are **not paid late**. 

- There are no **False Positives**, which means the model is reliable in its predictions regarding loans that are **not paid late**.

- However, the model has a **False Negative** count of 14, indicating that there are some loans that were **paid late** but were incorrectly classified as **not paid late**. This could be a concern in scenarios where accurately identifying loans that are likely to default is critical.

Overall, while the model performs well, further investigation into the **False Negatives** might be necessary to improve its predictive accuracy for **paid late** loans.

![Imgur](https://imgur.com/Nip0wLl.png)
![Imgur](https://imgur.com/cIESdCm.png)


## Cross-Validation of Random Forest Classifier

- After testing the Random Forest Classifier, i aimed to evaluate its performance using cross-validation to ensure the model's accuracy and robustness. Cross-validation helps to assess how the model generalizes to an independent dataset and can prevent overfitting.

### Steps Taken

1. **Data Preparation**:
   - Loaded the cleaned dataset (`normalized_data_for_ml.csv`).
   - Encoded the target variable (`paid_late`) using label encoding.
   - Separated features (`X`) and the target variable (`y`).

2. **Model Initialization**:
   - Initialized a Random Forest Classifier with 100 estimators.

3. **K-Fold Cross-Validation**:
   - Performed 5-fold cross-validation using `cross_val_score`.
   - Captured the accuracy scores across each fold.
     
![Imgur](https://imgur.com/o6yls9O.png)  

## Saving the Trained Model

After successfully training the Random Forest Classifier for loan default prediction, the model was saved for future use and testing. This allows for easy access to the trained model without needing to retrain it each time, which can be time-consuming and computationally expensive.

### Steps to Save the Model

1. **Model Training**: 
   The Random Forest Classifier was trained on the prepared dataset to predict whether a loan will be paid late.

2. **Saving the Model**: 
   The trained model was saved using the `joblib` library, which efficiently handles the serialization of Python objects. The command used to save the model is as follows:

   ```python
   import joblib
   joblib.dump(model, 'loan_default_prediction_model.pkl')

### Note
   - The Random Forest Classifier demonstrated improved performance compared to the standard Random Forest implementation when evaluated with cross-validation. Consequently, I will save the model trained using this optimized Random Forest Classifier for future use on test data.

---
## Training and validating the saved model (or making predictions) against the 2,000 loans in the test data.
- The same data cleaning and preprocessing carried out on the train_loan_data and train_payment_data was replicated on test_loan_data before using it on the saved model. 

## Model Evaluation on Test Data

This section outlines the process for evaluating predictions using the saved model `loan_default_prediction_model.pkl` on the `test_loan_data`.


## Evaluation Steps

1. **Import Necessary Libraries**: Import libraries such as `pandas`, `joblib`, and scikit-learn metrics for model evaluation.

2. **Load the Test Data**: The normalized test data is loaded from the CSV file `normalized_data_for_ml1.csv`.

3. **Separate Features and Target**: Features are extracted from the test dataset, and the true labels are stored separately.

4. **Load the Trained Model**:
   - Attempt to load the pre-trained model using `joblib`.
   - If the model is not found, the code will retrain a new model using the training data.

5. **Retraining the Model** (if necessary):
   - Load the training data from `normalized_data_for_ml.csv`.
   - Fit a Random Forest Classifier, adjusting hyperparameters if convergence warnings occur.
   - Monitor for overfitting by comparing training and test accuracies.

6. **Make Predictions**: Once the model is loaded or retrained successfully, predictions are made on the test dataset.

7. **Evaluate Model Performance**:
   - Calculate accuracy, confusion matrix, classification report, and AUC score.
   - Print the evaluation metrics for review.

8. **Save Predictions**: Predictions are added to the test data and saved to a new CSV file named `test_loan_predictions.csv1`.

## Metrics

- **Accuracy**: Measures the proportion of correct predictions out of the total predictions.
- **Confusion Matrix**: A summary of prediction results, showing true positive, false positive, true negative, and false negative counts.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **AUC Score**: The area under the ROC curve, indicating the model's ability to distinguish between classes.
  

## Predicted Outcome and Loan Repayment Prediction Summary

In this project, the goal was to predict whether a borrower would repay their loan on time or pay late. Below is a comparison between the actual loan repayment outcomes and the predictions made by the machine learning model.
- **Actual Percentage of People who Paid Late**: 8.60%
- **Actual Percentage of People who Paid on Time**: 91.40%
  
- **Predicted Percentage of People who Paid Late**: 11.35%
- **Predicted Percentage of People who Paid on Time**: 88.65%

## Insights:
- The model over-predicted the number of people who would pay late by about 2.75%. 
- It under-predicted the number of people who would pay on time by the same margin.
  
This suggests that while the model is generally accurate, it is slightly conservative in predicting late payments.

## Explanation of the Confusion Matrix Plot

- **True Positives (TP)**: The number of instances correctly predicted as 'Paid Late'.
  - Located in the bottom-right cell of the matrix.
  
- **True Negatives (TN)**: The number of instances correctly predicted as 'Not Paid Late'.
  - Located in the top-left cell of the matrix.
  
- **False Positives (FP)**: The number of instances incorrectly predicted as 'Paid Late' (Type I error).
  - Also known as a "false alarm".
  - Located in the top-right cell of the matrix.
  
- **False Negatives (FN)**: The number of instances incorrectly predicted as 'Not Paid Late' (Type II error).
  - Also known as a "missed detection".
  - Located in the bottom-left cell of the matrix.
 
  - This shows our model performed very well on the test data
 
    

![Imgur](https://imgur.com/uOgTZjK.png)
![Imgur](https://imgur.com/qIEqMlk.png)
![Imgur](https://imgur.com/hGQuC21.png)
![Imgur](https://imgur.com/JOV0kvi.png)
![Imgur](https://imgur.com/aAHhQfy.png)

## Loan Repayment Prediction Summary

In this project, the goal was to predict whether a borrower would repay their loan on time or pay late. Below is a comparison between the actual loan repayment outcomes and the predictions made by the machine learning model.


# Conclusion

The evaluation process provides a comprehensive analysis of the model's performance on unseen data, enabling further refinements if necessary.
## Model Performance Expectations

It's generally expected that the performance on the test data will be slightly lower than the performance on the training or validation data. This expectation arises from several reasons:

- **Overfitting**: The model might have learned the training data too well, including its noise and random variations. This can lead to lower accuracy when applied to new, unseen data.

- **Generalization**: The goal is for the model to generalize well to new data, not just memorize the training data. Test data helps us assess how well the model can generalize.

In the context of this work, I've already trained and evaluated the model using cross-validation and also on a validation set. The performance you see on the test data is a more realistic measure of how the model will perform in the real world when we deploy it to make predictions on new loan data.

If the test performance is significantly worse than the training or validation performance, it could indicate that the model is overfitting. You might need to consider strategies like:

- Reducing model complexity (e.g., using a simpler model or fewer features)
- Regularization techniques (e.g., L1 or L2 regularization)
- Increasing training data size
- Hyperparameter tuning (e.g., using grid search or random search)

- However, keep in mind that a slight drop in accuracy on the test data is often acceptable.











---



