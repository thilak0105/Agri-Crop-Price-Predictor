
'''
import pandas as pd

# Load the two CSV files
file1 = '/Users/thilak/PythonFiles/Crop price/DATA SET/2019.csv'
file2 = '/Users/thilak/PythonFiles/Crop price/DATA SET/2023.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine the two DataFrames row-wise (vertically)
combined_df = pd.concat([df1, df2], axis=1, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_file.csv', index=False)

print("Files have been combined successfully!")

'''
import pandas as pd
df = pd.read_csv('/Users/thilak/PythonFiles/Crop price/DATA SET/MAINDATA.csv')
df.info()
df.head()
