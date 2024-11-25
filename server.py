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

# @app.before_request
# def initialize():
#     if "initialized" not in session:
#         session["datasetChoice"] = "AirlineReviews"
#         session["modelType"] = "LogisticRegression"
#         session["initialized"] = True

#         g.a = Analyzer("AirlineReviews", "LogisticRegression", "text", "airline_sentiment")
#     else:
#         print("already initialized")

@app.route('/predict')
def predictText():
    datasetChoice = request.args.get('datasetChoice')
    modelType = request.args.get('modelType')
    text = request.args.get('text')

    result = {
        "datasetChoice": datasetChoice,
        "modelType": modelType,
        "text": text
    }

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