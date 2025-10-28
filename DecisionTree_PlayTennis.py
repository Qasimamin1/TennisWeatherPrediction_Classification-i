# ------------------------------------------------------------
# DecisionTree_PlayTennis Project
# ------------------------------------------------------------

# import all the libraries we need
# pandas for data handling, sklearn for machine learning, matplotlib for graph
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# read the dataset from the CSV file
# this file contains different weather conditions and whether we played tennis or not
df = pd.read_csv("weather.csv")

# show a few rows so we can confirm the data loaded correctly
print("Dataset preview:")
print(df.head())
print("------------------------------------------------")

# create label encoders to convert text data (like Sunny, Hot, etc.) into numbers
# the model only understands numeric values, not text
le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_hum = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

# apply encoding to each column of the dataset
# fit_transform learns the unique values and replaces them with numbers
df['Outlook'] = le_outlook.fit_transform(df['Outlook'])
df['Temperature'] = le_temp.fit_transform(df['Temperature'])
df['Humidity'] = le_hum.fit_transform(df['Humidity'])
df['Windy'] = le_wind.fit_transform(df['Windy'])
df['Play'] = le_play.fit_transform(df['Play'])

# print the encoded data to see the numeric version of our dataset
print("Encoded dataset:")
print(df.head())
print("------------------------------------------------")

# separate input features (X) and the target we want to predict (y)
# here X = weather conditions and y = play decision (Yes/No)
X = df[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = df['Play']

# create the Decision Tree model
# this model will learn patterns from the weather data to predict play decisions
model = DecisionTreeClassifier()

# train the model using the dataset
# during training, it builds decision rules like "If Outlook=Overcast -> Play=Yes"
model.fit(X, y)

print("Model training completed using CSV dataset.")
print("------------------------------------------------")

# test the trained model on the same dataset to see what it predicts
# this is just to verify the model has learned from the given data
predictions = model.predict(X)
print("Predictions on training data:")
print(predictions)
print("------------------------------------------------")

# now let's test the model with a new manual weather example
# this helps us see if the model can make predictions on unseen input
manual_data = pd.DataFrame([[
    le_outlook.transform(['Overcast'])[0],
    le_temp.transform(['Cool'])[0],
    le_hum.transform(['Normal'])[0],
    le_wind.transform([False])[0]
]], columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])

# make a prediction for this manual input
manual_pred = model.predict(manual_data)
manual_result = le_play.inverse_transform(manual_pred)

# display the manual input and what the model predicts for it
print("Manual test input -> Outlook=Overcast, Temp=Cool, Humidity=Normal, Windy=False")
print("Model Prediction from manual input:", manual_result[0])
print("------------------------------------------------")

# check which weather factors are most important for the decision
# higher value means that feature influences the decision more
print("Feature Importance:")
for name, score in zip(X.columns, model.feature_importances_):
    print(f"{name}: {score:.3f}")
print("------------------------------------------------")

# finally, draw the decision tree diagram
# each box represents a decision rule the model learned from the dataset
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=['No','Yes'], filled=True)
plt.title("Decision Tree - Play Tennis")
plt.show()
