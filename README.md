# Project Description
## Background
This machine learning project aims to predict the ecological footprint value of a country by considering factors such as the Human Development Index (HDI) and the geographical location of the continent where the country is located. The method used is linear regression with inputs such as HDI and categorical data representing continents. The output generated is the predicted ecological footprint value of a country. This project can provide insights into the factors contributing to environmental damage levels and can help in planning more sustainable environmental policies in the future.

**Objective**

Therefore, the objective of this final project is to build a machine learning-based model that can predict the ecological footprint value of a country's population using demographic inputs (such as HDI and life expectancy) and geographical features of a country (continent).

**Business Metrics**

The business metrics that can be derived include the effort and time required to analyze the sources contributing to increased ecological footprint in order to reduce the rate of environmental damage more efficiently and effectively.

## Project Architecture

This project aims to develop a machine learning model to predict a target variable. The process consists of the following steps:

1. **Data Preparation**

   In this step, the following activities will be carried out:

   - Data collection
   - Data definition, which includes determining the scope and limits of data values, including the range of values in numerical columns (HDI, EFConsPerCap), data types (str, int, float, etc.), and limits on classes in categorical columns (continent).
   - Data validation to ensure that each entry in the dataset complies with the defined data limits.
   - Data defense, a warning mechanism if there are data entries from the API that do not comply with the data definition.
   - Data splitting, dividing the dataset for training, testing, and validation purposes. The test size used is 20%.

2. **Exploratory Data Analysis (EDA)**

   The EDA phase is conducted to analyze whether the existing dataset has trends that could hinder or reduce model performance, and then analyze the appropriate modifications for the data. The analysis includes checking for null data, skewness, and data imbalance.

3. **Preprocessing**

   The processes in this stage include handling null data by imputation based on EDA results and feature engineering. For feature engineering, the processes include:

   - Transforming categorical data (continent) using one-hot encoding.
   - Data standardization.
   - Handling data imbalance by creating three datasets using different techniques: undersampling, oversampling, SMOTE.

4. **Modeling**

   In this stage, three regression models will be evaluated and one will be chosen as the production model: linear regression, Random Forest Regressor, and Decision Tree Regressor. The processes in this stage include:

   - Training and evaluating models, where at the end of this process, the best-performing model is determined based on MSE (lowest value), R-Square (highest value), and training time (shortest).
   - Optimization by performing hyper-parameter tuning, then repeating the training and model evaluation stages, and selecting the production model.
   - Documenting the results of training and model evaluation in the training log.

5. **Deployment**

   Deployment involves cloning the machine learning infrastructure from the host computer environment to a new environment to be accessible by other users and from anywhere using an API. Deployment in this final project will utilize Docker as a container on an AWS server.

## Expected Outputs

- ML prediction results from the regression model with the best performance.
- Training log recording:
   - Model performance metrics: Mean Squared Error (MSE), R-Square, and training time
   - Handling of data imbalance: undersampling, oversampling, SMOTE

# Documentation

## Data Format for API Prediction

1. Data format: This API requires input data in the form of a dictionary with two keys: "hdi" and "continent."

2. Data types: The "hdi" value must be numerical (float), while the "continent" value must be categorical data.

3. Maximum value for "hdi" is 1.0, and the minimum value is 0.0.
4. Accepted values for the "continent" key are: "Asia," "Europe," "Africa," "South America," "North America," and "Oceania."

5. Missing values: The API does not accept missing values. Ensure all values are present and valid.

## Prediction Format from API

- Predicted value of Ecological Footprint per Capita

## Workflow

![My Image](https://github.com/elfarahma/ml_process/blob/9a710b8c613ff38bddf6467da5364de29ecb18fa/ML_Process.jpg)


# Dataset Description

The dataset used in this project is a yearly time series dataset from 2000 to 2014, covering profile data of countries worldwide. The total entries in this dataset are 2156 data points. This dataset is a Github data repository processed by combining data from the World Bank and the Global Footprint Network (Shropshire, 2019).

The country profiles in this dataset include several variables:

- **Country**: covering 146 countries worldwide.
- **Continent**: covering 6 continental regions: Asia, Europe, Africa, South America, Oceania, and North America.
- **HDI (Human Development Index)**: a population parameter summarizing the well-being level of a region from aspects of health, education, and standard of living, on a scale of 0 to 1.
- **Life Expectancy**: average life expectancy of a country's population.
- **Population**: population of a country in a given year.
- **Ecological Footprint per Capita (EFConsPerCap)**: average environmental capacity per year spent per capita in a country for sustaining life, covering all primary, secondary, and tertiary needs (Global Hectare per capita).
- **Total Ecological Footprint in Global Hectares (EFConsTotGHA)**: total environmental capacity spent by a country (Total GHA).
- **Biocapacity per Capita (BiocapPerCap)**: a country's environmental capacity in providing resources to meet the lifestyle per capita (GHA per capita).
- **Total Biocapacity in Global Hectares (BiocapTotGHA)**: total environmental capacity of a country in providing resources for the lifestyle per capita (GHA per capita).

From the above dataset, the features used to predict the ecological footprint value of a country are:

- HDI ("hdi")
- Continent ("continent")

The ecological footprint parameter to be predicted is Ecological Footprint Per Capita (EFConsPerCap).

# How to Use
To use this machine learning model, you can send a POST request to the API endpoint with the following input payload:

