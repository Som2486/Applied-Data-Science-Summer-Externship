import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, header=None, names=columns)

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Records:")
print(df.head())

# Check for missing or zero values in the dataset
print("\nChecking for Zero Values:")
print((df == 0).sum())

# Replace zero values with NaN
df.replace(0, pd.NA, inplace=True)

# Fill missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Verify that there are no missing values
print("\nAfter Filling Missing Values:")
print(df.isnull().sum())

# Display the first few rows after preprocessing
print("\nFirst 5 Records After Preprocessing:")
print(df.head())
