To create and train the AI

1. Run create_model.py
    1a. Create the model
    1b. Save the model with a memorable name

2. Run Preprocessor.py
    2a. Upload breast-cancer-wisconsin.csv (class = diagnosis | 2 = benign | 4 = malignant)
    2b. Drop columns that aren't beneficial to predicting breast cancer (the ID column)
    2c. Get rid of any NaN values

3. Export data as "filename-preprocessed.csv"

4. Run train_model.py
    4a. Upload model.p
    4b. Upload training data
      i. Enter column headers
    4c. Choose what percentage of dataset to test accuracy with
    4d. Choose how many epochs or accuracy cutoff
