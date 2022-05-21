import pandas as pd
from Utils.PrepareData import clean
import joblib
from sklearn.metrics import accuracy_score

def TestModel(modelFilePath, csvTestFilePath, outputHeader):
    # Read the data from the input file
    data = clean(pd.read_csv(csv))

    # Get the output and input columns
    X = data.drop(columns=[output])
    y = data[output]

    # Read the model from file
    model = joblib.load(modelFilePath)

    # Test the model
    predictions = model.predict(X)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y, predictions)

    print("Accuracy: ", accuracy)

    return


if __name__ == '__main__':
    model = input("Enter the model file path: ")
    csv = input("Enter the csv test file path: ")
    output = input("Enter the output column name: ")

    TestModel(model, csv, output)