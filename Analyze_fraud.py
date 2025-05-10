import pandas as pd


# Load the dataset
df = pd.read_csv('credit_card_fraud.csv')

# Show first few rows
print(df.head())

# Check basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Class distribution
print(df['Fraud'].value_counts())

# Step 2: Preprocessing
# Make a copy of the dataframe
data = df.copy()

from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoders = {}

for column in ['User_ID', 'Location', 'Time', 'Device','Amount','Fraud']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Check the new data
print(data.head())

from sklearn.model_selection import train_test_split

# Step 3: Define features (X) and target (y)
X = data.drop(['Fraud', 'User_ID'], axis=1)  # Features (input)
y = data['Fraud']                             # Target (output)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Print the shape of the datasets
print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Step 4: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Now predict on the test data
y_pred = model.predict(X_test)

print("Model training complete!")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 5: Evaluate the model
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Detailed Report (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)