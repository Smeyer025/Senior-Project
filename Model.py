import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import re
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class Model:
    DATASETS = ["SocialMedia", "AmazonReviews", "TechProductReviews", "AirlineReviews", "DrugReviews", "HotelReviews", "MovieReviews", "News"]
    datasetChoice = ""
    MODEL_TYPES = ["LogisticRegression", "SupportVectorMachine", "RandomForest", "KMeansClustering"]
    modelType = ""
    df = pd.DataFrame()

    #Constructor
    def __init__(self, datasetChoice, modelType):
        self.datasetChoice = datasetChoice
        self.modelType = modelType
        self.df = pd.read_csv(f"{datasetChoice}.csv")
        print(self.df.head())

    #Generate model based on type of model chosen and dataset chosen
    #def generateModel(self):

    #Get info for the current dataset
    #def getDatasetInfo(self):
        
    def autoCleaner(self, textColumn, sentimentColumn, posLabel = "positive", negLabel = "negative"):
        self.df = self.df[[textColumn, sentimentColumn]]

        #Quanitify sentiment data
        def quantify_sentiment(a_col):
            if a_col == posLabel:
                return 1.0
            elif a_col == negLabel:
                return 0.0
            else:
                return 0.5
        
        #Transform ratings into the correct format (0.0-1.0)
        #def transform_numeric_ratings(a_col):
            
            
        #Is sentimentColumn already comprised of numbers? If not, change that!
        if not is_numeric_dtype(self.df[sentimentColumn]):
            self.df[sentimentColumn] = self.df[sentimentColumn].apply(quantify_sentiment)
        
        #Remove Usernames with @ from the text data 
        def remove_usernames(a_col):
            string = a_col
            usernamesAndHashtags = re.findall(r"(@\w+|#\w+)", a_col)
            for uAndH in usernamesAndHashtags:
                string = string.replace(uAndH, "")
            
            return string
        
        self.df[textColumn] = self.df[textColumn].apply(remove_usernames)
        print(self.df.head())
        print(self.df.tail())

            

        
model = Model("AirlineReviews", "LogisticRegression")
model.autoCleaner("text", "airline_sentiment")