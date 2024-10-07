import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix

class Model:
    #DATASETS = ["SocialMedia", "AmazonReviews", "TechProductReviews", "AirlineReviews", "DrugReviews", "HotelReviews", "MovieReviews", "News"]
    datasetChoice = ""
    #MODEL_TYPES = ["LogisticRegression", "SupportVectorMachine", "RandomForest", "KMeansClustering"]
    modelType = ""
    df = pd.DataFrame()
    model = "Model"
    x_train = ""
    x_test = ""
    y_train = ""
    y_test = ""
    vectorizer = ""
    x_train_vec = ""
    x_test_vec = ""
    y_pred = ""


    #Constructor
    def __init__(self, a_datasetChoice, a_modelType, a_textColumn, a_sentimentColumn, posLabel, negLabel, neutLabel):
        self.datasetChoice = a_datasetChoice
        self.modelType = a_modelType
        self.df = pd.read_csv(f"{a_datasetChoice}.csv")
        self.autoCleaner(a_textColumn, a_sentimentColumn, posLabel, negLabel, neutLabel)
        self.generateModel(a_textColumn, a_sentimentColumn)
        print(self.df.head())

    #Generate model based on type of model chosen and dataset chosen
    def generateModel(self, a_textColumn, a_sentimentColumn):
        #Split data into x and y values for Model
        x = self.df[a_textColumn]
        y = self.df[a_sentimentColumn]

        #Split data into training data (70%) and test data (30%)
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,test_size=0.3)

        #Make vectorizer, fit data to it
        self.vectorizer = CountVectorizer()
        self.x_train_vec = self.vectorizer.fit_transform(self.x_train)
        self.x_test_vec = self.vectorizer.transform(self.x_test)

        #Generate Model based on what was entered and fit it to the training data
        if self.modelType == "LogisticRegression":
            self.model = LogisticRegression(max_iter=350)
            self.model.fit(self.x_train_vec, self.y_train)
        elif self.modelType == "SupportVectorMachine":
            #Implement Later
            self.model = "svm"
        elif self.modelType == "RandomForest":
            #Implement Later
            self.model = "random forest"
        elif self.model == "KMeansClustering":
            #Implement Later
            self.model = "KMeansClustering"
        else:
            self.model = "UNSUPPORTED MODEL TYPE"
        
    def predict(self, textToPredict):
        textToPredict = [textToPredict]
        textToPredict_vec = self.vectorizer.transform(textToPredict)
        return self.model.predict(textToPredict_vec)

    #Get info for the current dataset
    #def getDatasetInfo(self):
        
    def autoCleaner(self, a_textColumn, a_sentimentColumn, posLabel="", negLabel="", neutLabel=""):
        self.df = self.df[[a_textColumn, a_sentimentColumn]]

        if is_numeric_dtype(self.df[a_sentimentColumn]):
            max = self.df[a_sentimentColumn].max()
            min = self.df[a_sentimentColumn].min()

            if min < 0:
                distToOne = 1 - min
                def no_negative(a_col):
                    return a_col + distToOne
                self.df[a_sentimentColumn] = self.df[a_sentimentColumn].apply(no_negative)
                min = min + distToOne
                max = max + distToOne

            def numeric_to_categorical(a_col):
                #print(f"a_col: {a_col}, max: {max}, min: {min}, a_col
                if a_col == max or a_col / max > 0.7:
                    return "positive"
                elif a_col == min or a_col / max < 0.4:
                    return "negative"
                else:
                    return "neutral"
            self.df[a_sentimentColumn] = self.df[a_sentimentColumn].apply(numeric_to_categorical)
            posLabel = "positive"
            negLabel = "negative"
            neutLabel = "neutral"
        
        #drop any labels that aren't specified
        labels = [f"{posLabel}", f"{negLabel}", f"{neutLabel}"]
        def drop_not_in_labels(a_col):
            if a_col not in labels:
                return np.nan
            else: 
                return a_col

        #drop any non-strings in text column
        def remove_not_str(a_col):
            if type(a_col) is not str:
                return np.nan
            else:
                return a_col

        self.df[a_sentimentColumn] = self.df[a_sentimentColumn].apply(drop_not_in_labels)
        self.df[a_textColumn] = self.df[a_textColumn].apply(remove_not_str)
        self.df = self.df.dropna()
        
        #Transform ratings into the correct format (0.0-1.0)
        #def transform_numeric_ratings(a_col):
            
            
        #Is a_sentimentColumn already comprised of numbers? If not, change that!
        # if not is_numeric_dtype(self.df[a_sentimentColumn]):
        #     self.df[a_sentimentColumn] = self.df[a_sentimentColumn].apply(quantify_sentiment)
        
        #Remove Usernames with @ from the text data 
        def clean_text(a_col):
            string = a_col
            usernamesAndHashtags = re.findall(r"(@\w+|#\w+)", a_col)
            for uAndH in usernamesAndHashtags:
                string = string.replace(uAndH, "")

            for word in string:
                if not word.isalnum():
                    word = re.sub(r'[\W_]+', "", word)
            
            return string
        
        stop = stopwords.words('english')
        def remove_stopwords(a_col):
            string = a_col.split()
            for word in string:
                for stopword in stop:
                    if word == stopword or word == stopword.upper():
                        string.remove(word)
                    
            putBack = ""
            for word in string:
                putBack = putBack + word + " "
            return putBack.strip()
        
        self.df[a_textColumn] = self.df[a_textColumn].apply(clean_text)
        self.df[a_textColumn] = self.df[a_textColumn].apply(remove_stopwords)
        return self.df
        print(self.df.head())
        print(self.df.tail())

model = Model("HotelReviews", "LogisticRegression", "reviews.text", "reviews.rating", "", "", "")
print(model.predict("This is awful"))
model.y_pred = model.model.predict(model.x_test_vec)
accuracy = accuracy_score(model.y_test, model.y_pred)
print("accuracy", accuracy)
heatmap = sns.heatmap(confusion_matrix(model.y_test,model.y_pred))
fig = heatmap.get_figure()
fig.savefig("out.png")