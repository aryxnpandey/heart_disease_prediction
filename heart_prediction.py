import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\hp\Desktop\HeartDiseasePrediction\heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Enter Patient Details")

age = int(input("Age: "))
sex = int(input("Sex (1=Male, 0=Female): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = int(input("Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (0/1): "))
restecg = int(input("Rest ECG (0-2): "))
thalach = int(input("Maximum Heart Rate: "))
exang = int(input("Exercise Angina (0/1): "))
oldpeak = float(input("Old Peak: "))
slope = int(input("Slope (0-2): "))
ca = int(input("CA (0-4): "))
thal = int(input("Thal (0-3): "))

patient = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]], columns=X.columns)

result = model.predict(patient)

probability = model.predict_proba(patient)

risk = probability[0][1] * 100

print(f"\nRisk Percentage: {risk:.2f}%")

if result[0] == 1:
    print("\nHeart Disease Detected")
else:
    print("\nNo Heart Disease Detected")