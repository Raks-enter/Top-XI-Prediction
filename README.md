# Cricket Top 11 Player Prediction

## Project Overview

In this project, we developed a machine learning application to predict the **Top 11 Performing Cricket Players** before a match. The goal is to assist coaches, analysts, and fans by forecasting player performances based on historical match data and statistics. This can help in making informed decisions about team selection and strategy.

## Problem Statement

Selecting the best cricket team before a match is crucial but challenging due to the many factors influencing player performance. Using machine learning, we aim to predict individual player performance scores using their past stats and then select the top 11 players with the highest predicted scores.

---

## Data Description

The dataset consists of historical player statistics with the following key columns:

- **Mat**: Number of matches played
- **Inns**: Number of innings played
- **NO**: Number of times not out
- **Runs**: Total runs scored
- **HS**: Highest score
- **Ave**: Batting average
- **BF**: Balls faced
- **SR**: Strike rate
- **100**: Number of centuries scored
- **50**: Number of half-centuries scored
- **0**: Number of ducks (zero runs scored)
- **4s**: Number of fours hit
- **6s**: Number of sixes hit

Some columns such as `Player` (player names), `Span` (career span), and `Unnamed: 0` (index column) were dropped as they are not numeric or useful for the prediction model.


## Step-by-Step Process

### 1. Data Ingestion

- We loaded the dataset (`odb.csv`) using Pandas.
- Removed irrelevant columns (`Player`, `Span`, `Unnamed: 0`) and cleaned the data.
- Split the data into training and testing sets using an 70:30 ratio.

### 2. Data Transformation

- Applied preprocessing steps such as scaling and normalization to bring all features onto a similar scale.
- Prepared feature arrays (`X`) and target variable (`y`) for model training.

### 3. Model Training

- Experimented with multiple regression models:
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Linear Regression
  - K-Nearest Neighbors
  - AdaBoost Regressor
- Used hyperparameter tuning (GridSearch) to find the best model parameters.
- Evaluated models on the test set using R² score.
- Selected the best performing model (with R² ≥ 0.6) and saved it as `model.pkl`.
- Also saved the preprocessing pipeline as `preprocessor.pkl` for consistent input scaling during prediction.

### 4. Prediction Pipeline

- Created a prediction pipeline that:
  - Loads the saved model and preprocessor.
  - Takes player stats as input.
  - Transforms the input using the preprocessor.
  - Predicts performance scores using the model.

### 5. Selecting Top 11 Players

- The prediction pipeline outputs a performance score for each player.
- From all predictions, the top 11 players with the highest predicted scores are selected as the predicted best team lineup.

### 6. Web Application

- Built a Flask web app with two main pages:
  - **Index page:** A welcome screen describing the project.
  - **Upload and Prediction page:** Allows users to upload a CSV file with player stats.
- On form submission:
  - The app reads the uploaded CSV.
  - Uses the prediction pipeline to compute predicted performance scores.
  - Displays the top 11 players with their predicted scores on a separate results page.

