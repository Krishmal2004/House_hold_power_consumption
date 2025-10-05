# Energy Consumption Data Preprocessing Pipeline

This project provides a comprehensive data preprocessing pipeline for cleaning, transforming, and preparing energy consumption datasets for analysis and machine learning modeling. The pipeline is demonstrated on three different energy-related datasets and includes steps for handling missing values, outlier removal, feature engineering, categorical variable encoding, and normalization/scaling.

## Datasets

The following datasets are used to showcase the preprocessing pipeline:

1. **Individual Household Electric Power Consumption**: This dataset contains measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. It is sourced from the UCI Machine Learning Repository.

2. **Appliances Energy Prediction**: This dataset contains experimental data used to create regression models of appliances energy use in a low energy building.

3. **Smart Home Energy Consumption**: This dataset contains energy consumption data from a smart home, including usage for various appliances and outdoor temperature readings.

## Preprocessing Pipeline

The pipeline consists of the following key steps, applied to each dataset:

### 1. Handling Missing Values
Missing values in the datasets are identified and handled. For numerical columns, missing values are replaced with the median, while for categorical columns, they are replaced with the mode.

### 2. Outlier Removal
Outliers are detected and removed using the Interquartile Range (IQR) method to prevent them from skewing the analysis and subsequent modeling.

### 3. Feature Engineering
New features are created from existing data:
- Converting date and time strings into datetime objects
- Extracting time-based features like hour, day_of_week, and month
- Creating additional meaningful features based on domain knowledge

### 4. Encoding Categorical Variables
Categorical variables are converted to numerical format:
- One-hot encoding for nominal categorical features
- Label encoding for ordinal categorical features

### 5. Normalization and Scaling
Numerical features are scaled to ensure they contribute equally to the analysis:
- MinMaxScaler is used to normalize features to a [0,1] range
- StandardScaler is used to standardize features to have zero mean and unit variance

## Exploratory Data Analysis (EDA)

Each dataset undergoes exploratory data analysis both before and after preprocessing:
- Distribution analysis using histograms and KDE plots
- Outlier visualization using box plots
- Relationship exploration using scatter plots
- Statistical summary of key metrics

## Results

The preprocessing pipeline successfully:
- Removes missing values and outliers
- Creates meaningful features from raw data
- Normalizes numerical features
- Encodes categorical variables
- Produces clean, analysis-ready datasets

### 6. Group Contribution

IT 24103866 - Obesekara S.O.K.D
IT 24103886 - Kapuge K.H.P.P
IT 24103938 - Premathilaka S.W.G.A.S
IT 24103952 - Harini R.
IT 24103976 - Bandara R.V.M.R.N
