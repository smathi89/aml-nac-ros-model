import pandas as pd

# Load the binary classification dataset
df = pd.read_csv("data/nac_peg_binary_classification.csv")

# Show how many samples are in each class
print("Class balance for 'ROS_reduced':")
print(df["ROS_reduced"].value_counts())
