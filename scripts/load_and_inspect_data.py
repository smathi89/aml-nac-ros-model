import pandas as pd

# Load your filled ROS dataset
df = pd.read_csv('data/combined_ros_measurements_filled.csv')

# Print shape (rows, columns)
print("Shape of dataset:", df.shape)

# Print missing values in each column
print("Missing values per column:")
print(df.isnull().sum())

# Preview first few rows
print("First 5 rows:")
print(df.head())
