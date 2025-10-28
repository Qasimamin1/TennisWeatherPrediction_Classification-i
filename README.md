TennisWeatherPrediction_Classification

Using Python machine learning project built in Python to predict whether we should play tennis based on weather conditions.
The model uses a Decision Tree Classifier from scikit-learn and is trained on a small CSV dataset containing different weather situations such as outlook, temperature, humidity, and wind.

Project Summary

The goal of this project was to understand the concept of classification in machine learning.
I created a dataset named weather.csv, trained a Decision Tree model, and then tested it on both:

The training dataset (CSV file)

A manual weather example that I entered directly in the code

This helped me see how the model makes decisions and how each weather factor affects the prediction.

Technologies Used

Python 3

pandas

scikit-learn

matplotlib

Visual Studio 2022

Steps Performed

Loaded the weather dataset from weather.csv.

Encoded all text data (Sunny, Rainy, etc.) into numeric form using LabelEncoder.

Trained the Decision Tree Classifier on the dataset.

Predicted results for both dataset and manual test input.

Displayed feature importance for each weather attribute.

Visualised the final decision tree using plot_tree() from sklearn.
