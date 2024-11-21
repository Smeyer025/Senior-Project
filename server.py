from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from Analyzer import Analyzer

app = Flask(__name__)

CORS(app)

@app.route('/server')
def predict():
    a = Analyzer("SocialMedia", "LogisticRegression", "clean_text", "category")
    return a.predict("I love this product!")

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

    a = ""
    if (datasetChoice == "SocialMedia"):
        a = Analyzer(datasetChoice, modelType, "clean_text", "category")
    elif (datasetChoice == "AirlineReviews"):
        a = Analyzer(datasetChoice, modelType, "text", "airline_sentiment")
    elif (datasetChoice == "DrugReviews"):
        a = Analyzer(datasetChoice, modelType, "review", "rating") 
    elif (datasetChoice == "HotelReviews"):
        a = Analyzer(datasetChoice, modelType, "reviews.text", "reviews.rating")
    elif (datasetChoice == "MovieReviews"):
        a = Analyzer(datasetChoice, modelType, "review", "sentiment")

    return jsonify(a.predict(text))

if __name__ == '__main__':
    app.run(debug=True, port=5000)