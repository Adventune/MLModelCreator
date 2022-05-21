""" Simple function to remove empty and duplicate rows from a dataframe """
def clean(data):
    # Prepare the data for the model
    data.replace("", float("NaN"), inplace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data