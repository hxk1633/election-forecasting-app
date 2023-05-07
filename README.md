# County-Level Election Forecasting

Web application allows users to build and analyze Bayesian models for forecasting election outcomes on the county-level. The front-end and back-end were built with Dash, a data science web application framework.

## User Interface
Users can select a state, multiple predictors, and election years for model calibration and testing. When the user clicks on the "Analyze" button as shown in the user interface below, the model uses a Bayesian modeling library called "Bambi" to build linear regreesion and logistic regression models. These models are used to forecast vote shares and party preferences for each county in the selected state. The model outputs consists of multiple graphs that visualize the range of possible election outcomes and the uncertainty of the model used to predict the election outcomes.

<img width="787" alt="user_interface" src="https://user-images.githubusercontent.com/34976129/236699069-57217fa7-ed76-48d3-8e2c-6c44e813a6a0.png">

## Project Setup

### Install dependencies
`pip install requirements.txt`

### Run Dash application
`python dash_app.py`
