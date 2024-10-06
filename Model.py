import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import re
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

class Model:
    #DATASETS = ["SocialMedia", "AmazonReviews", "TechProductReviews", "AirlineReviews", "DrugReviews", "HotelReviews", "MovieReviews", "News"]
    datasetChoice = ""
    #MODEL_TYPES = ["LogisticRegression", "SupportVectorMachine", "RandomForest", "KMeansClustering"]
    modelType = ""
    df = pd.DataFrame()

    #Constructor
    def __init__(self, a_datasetChoice, a_modelType, a_textColumn, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative"):
        self.datasetChoice = a_datasetChoice
        self.modelType = a_modelType
        self.df = pd.read_csv(f"{a_datasetChoice}.csv")
        print(self.autoCleaner(a_textColumn, a_sentimentColumn).head())

    #Generate model based on type of model chosen and dataset chosen
    def generateModel(self, a_modelType, a_textColumn, a_sentimentColumn):
        #Split data into x and y values for Model
        x = self.df[a_textColumn]
        y = self.df[a_sentimentColumn]

        #Split data into training data (70%) and test data (30%)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

    #Get info for the current dataset
    #def getDatasetInfo(self):
        
    def autoCleaner(self, a_textColumn, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative"):
        self.df = self.df[[a_textColumn, a_sentimentColumn]]
        
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
        
        self.df[a_textColumn] = self.df[a_textColumn].apply(clean_text)
        return self.df
        print(self.df.head())
        print(self.df.tail())

model = Model("AirlineReviews", "LogisticRegression", "text", "airline_sentiment")