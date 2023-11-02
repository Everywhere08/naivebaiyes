import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Create a DataFrame from the provided data
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)

# Initialize separate label encoders for each categorical variable
outlook_encoder = LabelEncoder()
temperature_encoder = LabelEncoder()
humidity_encoder = LabelEncoder()
wind_encoder = LabelEncoder()

# Fit and transform each variable
df["Outlook"] = outlook_encoder.fit_transform(df["Outlook"])
df["Temperature"] = temperature_encoder.fit_transform(df["Temperature"])
df["Humidity"] = humidity_encoder.fit_transform(df["Humidity"])
df["Wind"] = wind_encoder.fit_transform(df["Wind"])

# Split the data into features and target
X = df[["Outlook", "Temperature", "Humidity", "Wind"]]
y = df["PlayTennis"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
clf = GaussianNB()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Check the accuracy of the model
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Now, you can make predictions for new data entered by the user
new_data = {
    "Outlook": input("Enter Outlook (Sunny, Overcast, Rain): "),
    "Temperature": input("Enter Temperature (Hot, Mild, Cool): "),
    "Humidity": input("Enter Humidity (High, Normal): "),
    "Wind": input("Enter Wind (Weak, Strong): ")
}

# Transform new data using the respective label encoders
new_data["Outlook"] = outlook_encoder.transform([new_data["Outlook"]])[0]
new_data["Temperature"] = temperature_encoder.transform([new_data["Temperature"]])[0]
new_data["Humidity"] = humidity_encoder.transform([new_data["Humidity"]])[0]
new_data["Wind"] = wind_encoder.transform([new_data["Wind"]])[0]

prediction = clf.predict([[new_data["Outlook"], new_data["Temperature"], new_data["Humidity"], new_data["Wind"]]])
print(f"Predicted PlayTennis: {prediction[0]}")
