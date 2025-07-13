# Apartment Price Prediction

This project used **Ridge Regression** in machine learning to build a model explaining the prices of appartments based on the training sample and generate predictions for all observations from the test sample.

## Project Structure
```apartment-price-prediction/
├── data/
│    ├── appartments_train.csv
│    ├── appartments_test.csv
├── ridge_model.joblib
├── predictions.csv
├── apartments_regression.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

### Data features (columns)
The dataset consists of apartment records with the following features:

- `unit_id` – Unique (and anonymized) identifier for each apartment.
- `obj_type` – Type of apartment or object (categorical, anonymized).
- `dim_m2` – Apartment size in square meters.
- `n_rooms` – Number of rooms.
- `floor_no` – The floor on which the apartment is located.
- `floor_max` – Total number of floors in the building.
- `year_built` – The year the building was constructed.
- `dist_centre` – Distance from the apartment to the city center.
- `n_poi` – Number of points of interest nearby.
- `dist_sch` – Distance to the nearest school.
- `dist_clinic` – Distance to the nearest clinic.
- `dist_post` – Distance to the nearest post office.
- `dist_kind` – Distance to the nearest kindergarten.
`dist_rest` – Distance to the nearest restaurant.
`dist_uni` – Distance to the nearest college or university.
- `dist_pharma` – Distance to the nearest pharmacy.
- `own_type` – Ownership type (categorical, anonymized).
- `build_mat` – Building material (categorical, anonymized).
- `cond_class` – Condition or quality class of the apartment (categorical, anonymized).
- `has_park` – Whether the apartment has a parking space (boolean).
- `has_balcony` – Whether the apartment has a balcony (boolean).
- `has_lift` – Whether the apartment building has an elevator (boolean).
- `has_sec` – Whether the apartment has security features (boolean).
- `has_store` – Whether the apartment has a storage room (boolean).
- `price_z` – Target variable: Apartment price (in appropriate monetary units) to be predicted – only in the training sample
- `src_month` – Source month (time attribute).
- `loc_code` – Anonymized location code of the apartment.
- `market_volatility` – Simulated market fluctuation affecting the apartment price.
- `infrastructure_quality` – Indicator of the building’s infrastructure quality, partially based on the building’s age.
- `neighborhood_crime_rate` – Random index simulating local crime rate.
- `popularity_index` – Randomly generated measure of the apartment’s attractiveness.
- `green_space_ratio` – Proxy variable representing the amount of nearby green space, inversely related to the distance from the city center.
- `estimated_maintenance_cost` – Estimated cost of maintaining the apartment, based on its size.
- `global_economic_index` – Simulated economic index with minor fluctuations across entries, reflecting broader market conditions.

### Data files
appartments_train.csv – training data contains 156454 observations and 34 columns along with the target variable price_z.

appartments_test.csv – test data contains 39114 observations and 33 columns without the target variable.

### Steps
1. Exploratory Data Analysis
2. Feature Engineering
3. Modelling
4. Predictions

Different ML algorithms are compared through *cross-validation* and finally Ridge Regression is chosen to be applied due to its relatively higher R-squared values and lower RMSE value among all models, also it is one of the fastest model, hence it is used to fine-tune and finally to predict the final test data.

## Internal division of training data 
Internal division of trainng data is used to split data into train, validation and test samples to correctly assess performance of models on new data.

- X_train_internal, y_train_internal → For training and cross-validation
- X_val, y_val → For model selection & hyperparameter tuning.
- X_test_internal, y_test_internal → For final internal evaluation.
