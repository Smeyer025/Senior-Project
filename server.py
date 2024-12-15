##############################################################
# server.py - Creates backend endpoints to allow data to be  #
#             sent to the frontend to be displayed           #
##############################################################

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from Analyzer import Analyzer
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "ei3r988fe98j2nu90f90ioncxjnjwnljowdjoi"

CORS(app)

"""
makeReadable()

NAME
    makeReadable() - function that makes respnses from the Model class
                     more readable

SYNOPSIS
    String makeReadable(a_arr, a_map=True, a_heading="", a_matrix=False)
        a_arr     --> array response
        a_map     --> should labels be mapped to output?
        a_heading --> if there is a heading to output, add it to the string
        a_matrix  --> if a list of list, unspool it

DESCRIPTION
    This function makes a readable string out of the array of inputs passed
    as a_arr as well as the other parameters

RETURNS 
    Returns the readable string
"""
def makeReadable(a_arr, a_map=True, a_heading="", a_matrix=False):
    arrAsStr = f""
    if len(a_arr) == 2:
        labels = [Analyzer.ana.m_currPosLabel, Analyzer.ana.m_currNegLabel]
    else:
        labels = [Analyzer.ana.m_currPosLabel, Analyzer.ana.m_currNeutLabel, Analyzer.ana.m_currNegLabel]
    idx = 0

    for elem in a_arr:
        if a_matrix == False:
            if a_map:
                arrAsStr = arrAsStr + f"{labels[idx]}: {elem:.2f}, "
            else:
                arrAsStr = arrAsStr + f"{elem:.2f}, "
            idx = idx + 1
        else:
            arrAsStr = arrAsStr + f"["
            for subElem in elem:
                arrAsStr = arrAsStr + f"{subElem}, "
            arrAsStr = arrAsStr[:-2]
            arrAsStr = arrAsStr + f"]," + "\n"
        
    return (a_heading + arrAsStr[:-2])

"""
upload()

NAME
    upload() - function for handling file uploads

SYNOPSIS
    Response upload()
        HTTP args:
            file            --> File received from frontend
            textColumn      --> Label for text column in received file
            sentimentColumn --> Label for sentiment column in received file

DESCRIPTION
    This function receives the .csv file to be added as an option 
    in the datasets list, the text column in that dataset, and the 
    sentiment column in that dataset, saves the received file, 
    updates the list of datasets in the Analyzer class if that file
    has not already been imported, adds the column information to 
    the importedDatasetColumns dictionary in the Analyzer class, 
    and eiher returns the name of the file without the extension if
    the file is a .csv file or returns a message stating the .csv file
    requirement.     

RETURNS
    Returns the jsonified name of the uploaded file without the extension if 
    upload is a csv file, "Invalid filetype, upload .csv" otherwise
"""
@app.route('/uploadFile', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename.endswith(".csv"):
        file.save(os.path.join(os.getcwd(), f"{secure_filename(file.filename)}"))
        
        textColumn = request.args.get("text").strip('\"')
        sentimentColumn = request.args.get("sentiment").strip('\"')
        Analyzer.importedDatasetColumns[file.filename.split('.')[0]] = [textColumn, sentimentColumn]
        if file.filename.split('.')[0] not in Analyzer.DATASETS:
            Analyzer.DATASETS.append(file.filename.split('.')[0])
        print(Analyzer.DATASETS)

        return jsonify(file.filename.split('.')[0])
    else:
        return jsonify("Invalid filetype, upload .csv")

"""
predictText()

NAME
    predictText() - Function that handles model generation 
                    and predictions

SYNOPSIS
    Response predict()
        HTTP args:
            datasetChoice --> Dataset chosen for analysis
            modelType     --> Model chosen for analysis
            text          --> Text to be analyzed

DESCRIPTION
    This function takes in the choice of dataset, choice of 
    model, and the text sample, generates the model based on
    those specifications, and returns the jsonified prediction
    of the sentiment of that text sample.

RETURNS 
    Returns the jsonified prediction of the test sample's tone
"""
@app.route('/predict')
def predictText():
    datasetChoice = request.args.get('datasetChoice')
    modelType = request.args.get('modelType')
    text = request.args.get('text')

    # Analyzer.ana = Analyzer("AirlineReviews", "LogisticRegression", "text", "airline_sentiment")
    if (datasetChoice == "SocialMedia"):
        Analyzer.ana = Analyzer(datasetChoice, modelType, "clean_text", "category")
    elif (datasetChoice == "AirlineReviews"):
        Analyzer.ana = Analyzer(datasetChoice, modelType, "text", "airline_sentiment")
    elif (datasetChoice == "DrugReviews"):
        Analyzer.ana = Analyzer(datasetChoice, modelType, "review", "rating") 
    elif (datasetChoice == "HotelReviews"):
        Analyzer.ana = Analyzer(datasetChoice, modelType, "reviews.text", "reviews.rating")
    elif (datasetChoice == "MovieReviews"):
        Analyzer.ana = Analyzer(datasetChoice, modelType, "review", "sentiment")
    else:
        if datasetChoice in Analyzer.importedDatasetColumns.keys():
            Analyzer.ana = Analyzer(datasetChoice, modelType, Analyzer.importedDatasetColumns[datasetChoice][0], Analyzer.importedDatasetColumns[datasetChoice][1])
        else:
            raise Exception("Missing or invalid entered columns for this dataset")

    print("After: ", Analyzer.ana.m_currModelType)
    return jsonify(Analyzer.ana.predict(text))

"""
accuracy()

NAME
    accuracy() - Function that returns the accuracy of the model

SYNOPSIS
    Response accuracy()

DESCRIPTION
    This function returns the accuracy of the model

RETURNS
    Returns the jsonified accuracy of the model
"""
@app.route('/accuracy')
def accuracy():
    return jsonify(Analyzer.ana.m_currModel.accuracy())

"""
precision()

NAME
    precision() - Function that returns the precision of each class in the dataset

SYNOPSIS
    Response accuracy()

DESCRIPTION
    This function returns the precision of each class in the model

RETURNS
    Returns the jsonified precision of the model
"""
@app.route('/precision')
def precision():
    return jsonify(makeReadable(Analyzer.ana.m_currModel.precision().tolist()))

"""
recall()

NAME
    recall() - Function that returns the recall of each class in the dataset

SYNOPSIS
    Response recall()

DESCRIPTION
    This function returns the recall of each class in the model

RETURNS
    Returns the jsonified recall of the model
"""
@app.route('/recall')
def recall():
    return jsonify(makeReadable(Analyzer.ana.m_currModel.recall().tolist()))

"""
f1_score()

NAME
    f1_score() - Function that returns the f1 score of each class in the dataset

SYNOPSIS
    Response f1_score()

DESCRIPTION
    This function returns the f1 score of each class in the model

RETURNS
    Returns the jsonified f1 score of the model
"""
@app.route('/f1score')
def f1score():
    return jsonify(makeReadable(Analyzer.ana.m_currModel.f1_score().tolist()))

"""
hamming_loss()

NAME
    hamming_loss() - Function that returns the hamming loss of the model

SYNOPSIS
    Response hamming_loss()

DESCRIPTION
    This function returns the hamming loss of the model

RETURNS
    Returns the jsonified hamming loss of the model
"""
@app.route('/hamming_loss')
def hamming():
    return jsonify(Analyzer.ana.m_currModel.hamming_loss())

"""
kfold()

NAME
    kfold() - Function that runs k-fold cross validation
                        where k = 5

SYNOPSIS
    Response kfold()

DESCRIPTION
    This function runs k-fold cross validation where k = 5 and outputs the 
    Mean Squared Error for each fold.

RETURNS
    Returns the the jsonified Mean Squared Error for each fold
"""
@app.route('/kfold')
def kfold():
    return jsonify(makeReadable(Analyzer.ana.m_currModel.kfold().tolist(), a_map=False, a_heading="MSE: "))

"""
confusion_matrix()

NAME
    confusion_matrix() - Function that returns the confusion matrix of the model

SYNOPSIS
    Response confusion_matrix()

DESCRIPTION
    This function returns the confusion matrix of the model

RETURNS 
    Returns the jsonified confusion matrix of the model
"""
@app.route('/confusion_matrix')
def cm():
    return jsonify(makeReadable(Analyzer.ana.m_currModel.confusion_matrix().tolist(), a_heading="Confusion Matrix: ", a_matrix=True))

if __name__ == '__main__':
    app.run(debug=True, port=5000)