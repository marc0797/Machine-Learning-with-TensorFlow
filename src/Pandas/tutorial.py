# Short tutorial on how to use Pandas

import numpy as np
import pandas as pd

def main():
    # Create a DataFrame:
    #   Simple DataFrame containing 10 cells organized in 2 columns and 5 rows.
    #   Each column contains information about a person, the first column contains 
    #   the name of the person and the second column contains the age of the person.

    # Create and populate a 5x2 numpy array.
    my_data = np.array([['Tom', 10], ['Nick', 15], ['Juli', 14], ['Sam', 20], ['John', 25]])

    # Create a Python list with the column names.
    col_names = ['Name', 'Age']

    # Create a DataFrame from the numpy array and the column names.
    my_dataframe = pd.DataFrame(data=my_data, columns=col_names)

    # Print the DataFrame to the console.
    print("DataFrame:")
    print(my_dataframe, "\n")

    # Create a new column named height (in centimeters)
    my_dataframe['Height'] = [120, 150, 140, 170, 175]

    # Print the DataFrame to the console.
    print("DataFrame with new column:")
    print(my_dataframe, "\n")

    # Specify some subset of the DataFrame.
    print("Rows #0, #1 and #2:")
    print(my_dataframe.head(3), "\n")

    print("Row #2:")
    print(my_dataframe.iloc[2], "\n")

    print("Column 'Name':")
    print(my_dataframe['Name'], "\n")

    # Create a reference of the DataFrame.
    reference_to_my_dataframe = my_dataframe

    # Print the starting values of the DataFrame.
    print("  Starting value of DataFrame: %d" % my_dataframe["Height"][0])
    print("  Starting value of reference: %d\n" % reference_to_my_dataframe["Height"][0])

    # Change the value of the first row in the DataFrame.
    my_dataframe.at[0, "Height"] = 130
    print("  Updated value of DataFrame: %d" % my_dataframe["Height"][0])
    print("  Updated value of reference: %d\n\n" % reference_to_my_dataframe["Height"][0])

    # Create a true copy of the DataFrame.
    print("Creating a true copy of the DataFrame.")
    copy_of_my_dataframe = my_dataframe.copy()

    # Print the starting values of the DataFrame.
    print("  Starting value of DataFrame: %s" % my_dataframe["Age"][0])
    print("  Starting value of copy: %s\n" % copy_of_my_dataframe["Age"][0])

    # Change the value of the first row in the DataFrame.
    my_dataframe.at[0, "Age"] = 11
    print("  Updated value of DataFrame: %s" % my_dataframe["Age"][0])
    print("  Same value in the copy: %s\n" % copy_of_my_dataframe["Age"][0])



if __name__ == "__main__":
    main()