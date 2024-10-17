from flask import Flask
from Analyzer import Analyzer

app = Flask(__name__)

@app.route('/server')
def predict():
    a = Analyzer("SocialMedia", "LogisticRegression", "clean_text", "category")
    return a.predict("My Boyfriend is the cutest and most lovely man ever")

if __name__ == '__main__':
    app.run(debug=True)