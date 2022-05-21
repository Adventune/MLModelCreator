from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from Utils.PrepareData import clean

def createDesicionTreeModel(inputFile, outputDataHeader ):
    # Read the data from the input file
    data = clean(pd.read_csv(inputFile))

    # Get the output and input columns
    X = data.drop(columns=[outputDataHeader])
    y = data[outputDataHeader]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = DecisionTreeClassifier()
    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: ", accuracy)

    return model


if(__name__ == '__main__'):
    inputFile = input("Enter the input file path: ")
    outputDataHeader = input("Enter the name of the output data header: ")

    createDesicionTreeModel(inputFile, outputDataHeader)


