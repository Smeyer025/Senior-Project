##############################################################
# Model Class - Allows users to create machine learning      #
#               models on the fly based on chosen parameters #
##############################################################

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
    datasetChoice = ""
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

    """
    Model::__init__()

    NAME
        Model::__init__() - Constructor for Model Class
    
    SYNOPSIS
        void __init__(self, a_datasetChoice, a_modelType, a_textColumn, a_sentimentColumn, a_posLabel, a_negLabel, a_neutLabel, a_balance)
            self              --> Required first parameter for any Python class function/constructor: the object being created/operated on
            a_datasetChoice   --> Dataset chosen for the model
            a_modelType       --> Model type chosen for the model
            a_textColumn      --> Column in dataset containing text corresponding to sentiment rating
            a_sentimentColumn --> Column in dataset containing sentiment rating corresponding to text
            a_posLabel        --> Label denoting positive tone
            a_negLabel        --> Label denoting negative tone
            a_neutLabel       --> Label denoting neutral tone
            a_balance         --> Boolean containing whether or not the dataset should be balanced as part of data cleaning

    DESCRIPTION
        This constructor takes in all necessary information for the data to be cleaned and for the model to be created,
        assigns all necessary member variables, and runs the autoCleaner and generateModel functions.
    """
    def __init__(self, a_datasetChoice, a_modelType, a_textColumn, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative", a_neutLabel = "neutral", a_balance = False):
        self.datasetChoice = a_datasetChoice
        self.modelType = a_modelType
        self.df = pd.read_csv(f"{a_datasetChoice}.csv")
        self.autoCleaner(a_textColumn, a_sentimentColumn, a_posLabel, a_negLabel, a_neutLabel, a_balance)
        self.generateModel(a_textColumn, a_sentimentColumn)

    """
    Model::generateModel()

    NAME
        Model::generateModel() - Function that handles ML model generation
    
    SYNOPSIS
        void generateModel(self, a_textColumn, a_sentimentColumn)
            self              --> Required first parameter for any Python class function/constructor. The object being created/operated on 
            a_textColumn      --> Column in dataset containing text corresponding to sentiment rating
            a_sentimentColumn --> Column in dataset containing sentiment rating corresponding to text

    DESCRIPTION
        This function takes in the names of the text and sentiment columns, splits those into two separate series x and y,
        splits the data into training and test data, uses a count vectorizer to determine the value of the different words
        present in the text, and creates a model based on whichever model type is chosen in the constructor.
    """
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
        elif self.model == "KNearestNeighbors":
            #Implement Later
            self.model = "KNearestNeighbors"
        else:
            self.model = "UNSUPPORTED MODEL TYPE"
        
    """
    Model::predict()

    NAME
        Model::predict() - Function that takes in text to predict and returns the model's prediction 
                           of that text's sentiment.
    
    SYNOPSIS
        String predict(self, a_textToPredict)
            self              --> Required first parameter for any Python class function/constructor. The object being created/operated on 
            a_textToPredict   --> Text to be predicted

    DESCRIPTION
        This function takes in the text whose sentiment is to be predicted, fits it to the vectorizer 
        created in the constructor, uses the model to predict the text's sentiment, and returns that prediction.
    """
    def predict(self, a_textToPredict):
        a_textToPredict = [a_textToPredict]
        textToPredict_vec = self.vectorizer.transform(a_textToPredict)
        return self.model.predict(textToPredict_vec)

    """
    Model::balance()

    NAME
        Model::balance() - Function that balances the spread of data in the dataset
    
    SYNOPSIS
        DataFrame balance(self, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative", a_neutLabel = "neutral")
            self              --> Required first parameter for any Python class function/constructor: the object being created/operated on
            a_sentimentColumn --> Column in dataset containing sentiment rating corresponding to text
            a_posLabel        --> Label denoting positive tone
            a_negLabel        --> Label denoting negative tone
            a_neutLabel       --> Label denoting neutral tone

    DESCRIPTION
        This function takes in the sentiment column, the positive label, the negative label, and the neutral
        label, finds whichever of the three has the least prevalence in the dataset, and randomly 
        removes features with the other two sentiments until all three have the same amount of features
        in the dataset. The new dataset, with the balanced spread, is returned.
    """
    def balance(self, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative", a_neutLabel = "neutral"):
        nums = {"neutral": len(self.df[self.df[a_sentimentColumn] == a_neutLabel].index), "positive": len(self.df[self.df[a_sentimentColumn] == a_posLabel].index), "negative": len(self.df[self.df[a_sentimentColumn] == a_negLabel].index)}
        indices = {"neutral": self.df[self.df[a_sentimentColumn] == a_neutLabel].index, "positive": self.df[self.df[a_sentimentColumn] == a_posLabel].index, "negative": self.df[self.df[a_sentimentColumn] == a_negLabel].index}
        dfs = {"neutral": pd.DataFrame(self.df.loc[indices["neutral"]]), "positive": pd.DataFrame(self.df.loc[indices["positive"]]), "negative": pd.DataFrame(self.df.loc[indices["negative"]])}

        for key in nums.keys():
            remove_n = -(min(nums.values()) - nums[key])
            if remove_n != 0:
                drop_indices = np.random.choice(indices[key], remove_n, replace=False)
                dfs[key] = dfs[key].drop(drop_indices)
        return pd.concat([dfs[a_posLabel], dfs[a_neutLabel], dfs[a_negLabel]])
        
    """
    Model::autoCleaner()

    NAME
        Model::autoCleaner() - Function that cleans the data for model generation
    
    SYNOPSIS
        DataFrame autoCleaner(self, a_textColumn, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative", a_neutLabel = "neutral", a_balance = False)
            self              --> Required first parameter for any Python class function/constructor: the object being created/operated on
            a_textColumn      --> Column in dataset containing text corresponding to sentiment rating
            a_sentimentColumn --> Column in dataset containing sentiment rating corresponding to text
            a_posLabel        --> Label denoting positive tone
            a_negLabel        --> Label denoting negative tone
            a_neutLabel       --> Label denoting neutral tone
            a_balance         --> Boolean containing whether or not the dataset should be balanced as part of data cleaning

    DESCRIPTION
        This function takes in the text column, the sentiment column, the labels names, and whether or not 
        the data should be balanced, and cleans the data. Steps to data cleaning are:
            1. Drop all columns except for the text column and the sentiment column
            2. Check the sentiment column's dataset data type. If numeric, convert to categorical
               by identifying the minimum and maximum in the range of the column, assigning negative
               to the bottom 33%, neutral to text between 34% and 66%, and positive to the top 33%.
               If data has been updated in this way, assign "positive" to a_posLabel, "negative" to
               a_negLabel, and "neutral" to a_neutLabel.
            3. Drop any features with sentiment labels not matching a_posLabel, a_negLabel, or a_neutLabel.
            4. Drop any features with non-string data in the text column
            5. Remove any hashtags (words starting with '#') or usernames (words starting with '@')
            6. Remove any words with non-alphanumeric characters
            7. Remove stopwords (common toneless words such as I, are, they, etc.).
            8. If specified in a_balance, balance dataset
        Returns self.df
    """
    def autoCleaner(self, a_textColumn, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative", a_neutLabel = "neutral", a_balance = False):
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
                if a_col == max or a_col / max >= 0.67:
                    return "positive"
                elif a_col == min or a_col / max <= 0.33:
                    return "negative"
                else:
                    return "neutral"
            self.df[a_sentimentColumn] = self.df[a_sentimentColumn].apply(numeric_to_categorical)
            a_posLabel = "positive"
            a_negLabel = "negative"
            a_neutLabel = "neutral"
        
        #drop any labels that aren't specified
        labels = [f"{a_posLabel}", f"{a_negLabel}", f"{a_neutLabel}"]
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
        
        #Remove Usernames with @ from the text data, hashtags
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
        if a_balance:
            self.df = self.balance(a_sentimentColumn, a_posLabel, a_negLabel, a_neutLabel)
        return self.df
    
    """
    Model::test()

    NAME
        Model::test() - Test function for the model class
    
    SYNOPSIS
        String test(self, a_textToPredict)
            self              --> Required first parameter for any Python class function/constructor. The object being created/operated on 
            a_textToPredict   --> Text to be predicted

    DESCRIPTION
        This function  takes in text to predict, generates a model, runs and prints a prediction on 
        the inputted text, outputs information about the model, and creates and saves a heatmap
        of the model's confusion matrix.
    """
    def test(self, a_textToPredict):
        model = Model("SocialMedia", "LogisticRegression", "clean_text", "category")
        print(model.predict(a_textToPredict))
        model.y_pred = model.model.predict(model.x_test_vec)
        accuracy = accuracy_score(model.y_test, model.y_pred)
        print("accuracy", accuracy)
        heatmap = sns.heatmap(confusion_matrix(model.y_test,model.y_pred))
        fig = heatmap.get_figure()
        fig.savefig("out.png")
