from ModelBuild.DesicionTreeBuild import createDesicionTreeModel 
import joblib

pathToInputFile = "sampleData.csv"
pathToOutputFile = "model.joblib"
outputDataHeader = 'column_1'

decisionTreeModel = createDesicionTreeModel(pathToInputFile, outputDataHeader)
joblib.dump(decisionTreeModel, pathToOutputFile)