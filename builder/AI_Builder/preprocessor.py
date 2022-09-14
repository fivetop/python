#Preprocessor.py
import os
import pickle
import tqdm
import pandas as pd
import numpy as np

os.system('clear')
#Load CSV
print('What is the name of the CSV file you would like to upload? If it is not already in the directory, please place it there now. Enter the name of the CSV with the .csv extension.')
file_name = raw_input('> ')

df = pd.read_csv(file_name)
print('\nFile loaded successfuly.')

#Ask to drop columns
print('Are there any COLUMNS in the datafile that should not be included in the training and should be dropped? (Y/N)')
in1 = raw_input('> ')
if in1 == 'Y':
    print('What is the label of the column should be dropped?')
    in2 = raw_input('> ')
    df.drop(in2, axis=1, inplace=True)
    dropping_col = True
    while(dropping_col):
        print('Are there any other columns that should be dropped? (Y/N)')
        in3 = raw_input('> ')
        if in3 == 'Y':
            print('What is the label of the column should be dropped?')
            in2 = raw_input('> ')
            df.drop(in2, axis=1, inplace=True)
            dropping_col = True
        else:
            dropping_col = False

#Ask to drop rows
print('Are there any ROWS in the datafile that are outliers and should be dropped? (Y/N)')
in1 = raw_input('> ')
if in1 == 'Y':
    print('What is the index of the row should be dropped?')
    in2 = raw_input('> ')
    df.drop(df.index[in2], inplace=True)
    dropping_row = True
    while(dropping_row):
        print('Are there any other rows that should be dropped? (Y/N)')
        in3 = raw_input('> ')
        if in3 == 'Y':
            print('What is the index of the column should be dropped?')
            in2 = raw_input('> ')
            df.drop(df.index[in2], inplace=True)
            dropping_row = True
        else:
            dropping_row = False

#Remove holes in dataset
print('Are there any holes in the data that need to be removed? (Y/N)')
in1 = raw_input('> ')
if in1 == 'Y':
    print('How are the holes denoted? (i.e. "?", " ", etc.)')
    in2 = raw_input('> ')
    df.replace(in2, -99999, inplace=True)
    dropping_NaN = True
    while(dropping_NaN):
        print('Are there any other holes that should be dropped? (Y/N)')
        in3 = raw_input('> ')
        if in3 == 'Y':
            print('What is the placeholder of the holes that should be dropped?')
            in2 = raw_input('> ')
            df.replace(in2, -99999, inplace=True)
            dropping_NaN = True
        else:
            dropping_NaN = False

#Convert strings to integers
print('Are there any strings in the dataset? (Y/N)')
in4 = raw_input('> ')
if in4 == 'Y':
    print('What column label are the strings in?')
    string_col_label = raw_input('> ')

    columns = df.columns.values
    for column in columns:
        text_digits_vals = {}
        def convert_to_int(val):
            return text_digits_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digits_vals:
                    text_digits_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    convert_strings = True
    while(convert_strings):
        print('Are there any other columns with strings? (Y/N)')
        in3 = raw_input('> ')
        if in3 == 'Y':
            pstring_col_label = raw_input('> ')

            columns = df.columns.values
            for column in columns:
                text_digits_vals = {}
                def convert_to_int(val):
                    return text_digits_vals[val]

                if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                    column_contents = df[column].values.tolist()
                    unique_elements = set(column_contents)
                    x = 0
                    for unique in unique_elements:
                        if unique not in text_digits_vals:
                            text_digits_vals[unique] = x
                            x+=1

                    df[column] = list(map(convert_to_int, df[column]))
            convert_strings = True
        else:
            convert_strings = False

df.to_csv("{}-processed.csv".format(file_name), index=False)
print("Pre-Processed file saved successfuly")

#Testing percentage  <==== Do this later
#print('What percentage of the dataset should be used for testing the accuracy of the model?')
