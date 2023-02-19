# Heart Disease Prediction
This project is focused on predicting the risk of heart disease based on various clinical and demographic features of patients. The dataset used in this project contains information on patients who were tested for heart disease and whether they tested positive or negative. The goal of this project is to build a model that can accurately predict the risk of heart disease in new patients based on their features.

## Project Overview
The project consists of the following tasks:

- Preliminary Analysis
- Perform preliminary data inspection and report the findings on the structure of the data, missing values, duplicates, etc.
- Based on these findings, remove duplicates (if any) and treat missing values using an appropriate strategy.
- Data Exploration and Analysis
- Get a preliminary statistical summary of the data and explore the measures of central tendencies and spread of the data.
- Identify the data variables which are categorical and describe and explore these variables using appropriate tools, such as count plot.
- Study the occurrence of CVD (Cardiovascular Disease) across the Age category.
- Study the composition of all patients with respect to the Sex category.
- Study if one can detect heart attacks based on anomalies in the resting blood pressure (trestbps) of a patient.
- Describe the relationship between cholesterol levels and the target variable.
- State what relationship exists between peak exercising and the occurrence of a heart attack.
- Check if thalassemia is a major cause of CVD.
- List how the other factors determine the occurrence of CVD.
- Use a pair plot to understand the relationship between all the given variables.
- Build a baseline model to predict the risk of a heart attack using logistic regression and random forest and explore the results while using correlation analysis and logistic regression (leveraging standard error and p-values from statsmodels) for feature selection.

## Dataset
- The dataset used in this project contains 14 variables, including the target variable, which is the presence or absence of heart disease. The variables in the dataset include demographic and clinical features such as age, sex, blood pressure, cholesterol level, and ECG results.

## Requirements
- This project was implemented using Python 3.7 and several Python libraries, including Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Statsmodels. To run the code and reproduce the results, these libraries should be installed in your local environment.

## Results
- The results of the project are summarized in the Jupyter Notebook file heart_disease_prediction.ipynb. The notebook includes the code and output for each of the tasks listed above. The final baseline model was built using logistic regression and random forest, and correlation analysis and logistic regression were used for feature selection. The model achieved an accuracy of 84% in predicting the risk of heart disease, which is a reasonable starting point for further model development and refinement.

## Conclusion
- This project demonstrated the application of machine learning techniques for predicting the risk of heart disease based on clinical and demographic features of patients. The results showed that certain features such as age, sex, blood pressure, and cholesterol level are strong predictors of the risk of heart disease. Future work can focus on refining the model and improving its accuracy, as well as exploring other machine learning algorithms and feature selection techniques.
