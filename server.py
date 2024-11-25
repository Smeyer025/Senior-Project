from flask import Flask
from flask import request
from flask import jsonify
from flask import session
from flask import g
from flask_cors import CORS
from Analyzer import Analyzer
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "ei3r988fe98j2nu90f90ioncxjnjwnljowdjoi"

CORS(app)

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
    Response upload()
        HTTP args:
            datasetChoice --> Dataset chosen for analysis
            modelType     --> Model chosen for analysis
            text          --> Text to be analyzed

DESCRIPTION
    This function takes in the choice of dataset, choice of 
    model, and the text sample, generates the model based on
    those specifications, and returns the jsonified prediction
    of the sentiment of that text sample.
"""
@app.route('/predict')
def predictText():
    datasetChoice = request.args.get('datasetChoice')
    modelType = request.args.get('modelType')
    text = request.args.get('text')

    # if("initialized" in session):
    #     if(session["modelType"] != modelType or session["datasetChoice"] != datasetChoice):
    #         print("previous: " + session["modelType"])
    #         print("previous: " + session["datasetChoice"])
    #         print("new: " + modelType)
    #         print("new: " + datasetChoice)
    #         print("current: " + g.a.currModelType)

    #         session["modelType"] = modelType
    #         session["datasetChoice"] = datasetChoice
    #         session["initialized"] = True

    if (datasetChoice == "SocialMedia"):
        g.a = Analyzer(datasetChoice, modelType, "clean_text", "category")
    elif (datasetChoice == "AirlineReviews"):
        g.a = Analyzer(datasetChoice, modelType, "text", "airline_sentiment")
    elif (datasetChoice == "DrugReviews"):
        g.a = Analyzer(datasetChoice, modelType, "review", "rating") 
    elif (datasetChoice == "HotelReviews"):
        g.a = Analyzer(datasetChoice, modelType, "reviews.text", "reviews.rating")
    elif (datasetChoice == "MovieReviews"):
        g.a = Analyzer(datasetChoice, modelType, "review", "sentiment")
    else:
        if datasetChoice in Analyzer.importedDatasetColumns.keys():
            g.a = Analyzer(datasetChoice, modelType, Analyzer.importedDatasetColumns[datasetChoice][0], Analyzer.importedDatasetColumns[datasetChoice][1])
        else:
            raise Exception("Missing or invalid entered columns for this dataset")

    return jsonify(g.a.predict(text))

if __name__ == '__main__':
    app.run(debug=True, port=5000)