##############################################################
# Model Class - Allows users to create machine learning      #
#               models on the fly based on chosen parameters #
##############################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

"""
Model

NAME
    Model: Model class; Allows for on the fly generation of ML models 
                        for the purpose of sentiment analysis

SYNOPSIS
    __init__(): Constructor
    generateModel(): Create new ML model
    predict(): Use model to make a prediction 
    balance(): Remove over-represented groups at random until dataset is balanced
    autoCleaner(): Clean dataset, remove unnecessary columns, remove non-words


DESCRIPTION
    This class provides functionality that controls the Model class and ensures
    the frontend can access it easily.
"""
class Model:
    m_datasetChoice = ""
    m_modelType = ""
    m_df = pd.DataFrame()
    m_model = "Model"
    m_x = ""
    m_y = ""
    m_x_train = ""
    m_x_test = ""
    m_y_train = ""
    m_y_test = ""
    m_vectorizer = ""
    m_x_vec = ""
    m_x_train_vec = ""
    m_x_test_vec = ""
    m_y_pred = ""

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
        self.m_datasetChoice = a_datasetChoice
        self.m_modelType = a_modelType
        self.m_df = pd.read_csv(f"{a_datasetChoice}.csv")
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
        self.m_x = self.m_df[a_textColumn]
        self.m_y = self.m_df[a_sentimentColumn]

        #Split data into training data (70%) and test data (30%)
        self.m_x_train,self.m_x_test,self.m_y_train,self.m_y_test=train_test_split(self.m_x,self.m_y,test_size=0.3)

        #Make vectorizer, fit data to it
        self.m_vectorizer = CountVectorizer()
        self.m_x_vec = self.m_vectorizer.fit_transform(self.m_x)
        self.m_x_train_vec = self.m_vectorizer.fit_transform(self.m_x_train)
        self.m_x_test_vec = self.m_vectorizer.transform(self.m_x_test)

        print("modelType: " + self.m_modelType)

        #Generate Model based on what was entered and fit it to the training data
        if self.m_modelType == "LogisticRegression":
            self.m_model = LogisticRegression(max_iter=350)
            self.m_model.fit(self.m_x_train_vec, self.m_y_train)
        elif self.m_modelType == "SupportVectorMachine":
            #Implement Later
            self.m_model = svm.SVC()
            self.m_model.fit(self.m_x_train_vec, self.m_y_train)
        elif self.m_modelType == "RandomForest":
            #Implement Later
            self.m_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.m_model.fit(self.m_x_train_vec, self.m_y_train)
        elif self.m_modelType == "KNearestNeighbors":
            #Implement Later
            self.m_model = KNeighborsClassifier(n_neighbors=3)
            self.m_model.fit(self.m_x_train_vec, self.m_y_train)
        elif self.m_modelType == "VotingClassifier":
            lr = LogisticRegression(max_iter=350)
            svc = svm.SVC()
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            KNN = KNeighborsClassifier(n_neighbors=3)
            self.m_model = VotingClassifier(estimators=[('lr', lr), ('svc', svc), ('rf', rf), ('knn', KNN)])
            self.m_model.fit(self.m_x_train_vec, self.m_y_train)
        else:
            self.m_model = "UNSUPPORTED MODEL TYPE"
        
    """
    Model::predict()

    NAME
        Model::predict() - Function that takes in text to predict and returns the model's prediction 
                           of that text's sentiment.
    
    SYNOPSIS
        list predict(self, a_textToPredict)
            self              --> Required first parameter for any Python class function/constructor. The object being created/operated on 
            a_textToPredict   --> Text to be predicted

    DESCRIPTION
        This function takes in the text whose sentiment is to be predicted, fits it to the vectorizer 
        created in the constructor, uses the model to predict the text's sentiment, and returns that prediction.

    RETURNS
        Returns a list with the String prediction of positive, negative, or neutral inside
    """
    def predict(self, a_textToPredict):
        a_textToPredict = [a_textToPredict]
        textToPredict_vec = self.m_vectorizer.transform(a_textToPredict)
        return list(self.m_model.predict(textToPredict_vec))

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

    RETURNS
        Returns the updated dataset
    """
    def balance(self, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative", a_neutLabel = "neutral"):
        nums = {"neutral": len(self.m_df[self.m_df[a_sentimentColumn] == a_neutLabel].index), "positive": len(self.m_df[self.m_df[a_sentimentColumn] == a_posLabel].index), "negative": len(self.m_df[self.m_df[a_sentimentColumn] == a_negLabel].index)}
        indices = {"neutral": self.m_df[self.m_df[a_sentimentColumn] == a_neutLabel].index, "positive": self.m_df[self.m_df[a_sentimentColumn] == a_posLabel].index, "negative": self.m_df[self.m_df[a_sentimentColumn] == a_negLabel].index}
        dfs = {"neutral": pd.DataFrame(self.m_df.loc[indices["neutral"]]), "positive": pd.DataFrame(self.m_df.loc[indices["positive"]]), "negative": pd.DataFrame(self.m_df.loc[indices["negative"]])}

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
    
    RETURNS 
        Returns the dataframe post-cleaning
    """
    def autoCleaner(self, a_textColumn, a_sentimentColumn, a_posLabel = "positive", a_negLabel = "negative", a_neutLabel = "neutral", a_balance = False):
        self.m_df = self.m_df[[a_textColumn, a_sentimentColumn]]

        if is_numeric_dtype(self.m_df[a_sentimentColumn]):
            max = self.m_df[a_sentimentColumn].max()
            min = self.m_df[a_sentimentColumn].min()

            if min < 0:
                distToOne = 1 - min
                def no_negative(a_col):
                    return a_col + distToOne
                self.m_df[a_sentimentColumn] = self.m_df[a_sentimentColumn].apply(no_negative)
                min = min + distToOne
                max = max + distToOne

            def numeric_to_categorical(a_col):
                if a_col == max or a_col / max >= 0.67:
                    return "positive"
                elif a_col == min or a_col / max <= 0.33:
                    return "negative"
                else:
                    return "neutral"
            self.m_df[a_sentimentColumn] = self.m_df[a_sentimentColumn].apply(numeric_to_categorical)
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

        self.m_df[a_sentimentColumn] = self.m_df[a_sentimentColumn].apply(drop_not_in_labels)
        self.m_df[a_textColumn] = self.m_df[a_textColumn].apply(remove_not_str)
        self.m_df = self.m_df.dropna()
        
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
        
        self.m_df[a_textColumn] = self.m_df[a_textColumn].apply(clean_text)
        self.m_df[a_textColumn] = self.m_df[a_textColumn].apply(remove_stopwords)
        if a_balance:
            self.m_df = self.m_balance(a_sentimentColumn, a_posLabel, a_negLabel, a_neutLabel)
        return self.m_df

    """
    Model::accuracy()

    NAME
        Model::accuracy() - Function that returns the accuracy of the model
    
    SYNOPSIS
        float accuracy()

    DESCRIPTION
        This function returns the accuracy of the model

    RETURNS
        Returns the accuracy of the model
    """
    def accuracy(self):
        self.m_y_pred = self.m_model.predict(self.m_x_test_vec)
        return accuracy_score(self.m_y_test, self.m_y_pred)
    
    """
    Model::precision()

    NAME
        Model::precision() - Function that returns the precision of each class in the dataset
    
    SYNOPSIS
        list accuracy()

    DESCRIPTION
        This function returns the precision of each class in the model

    RETURNS
        Returns the precision of the model
    """
    def precision(self):
        self.m_y_pred = self.m_model.predict(self.m_x_test_vec)
        return precision_score(self.m_y_test, self.m_y_pred, average=None)
    
    """
    Model::recall()

    NAME
        Model::recall() - Function that returns the recall of each class in the dataset
    
    SYNOPSIS
        list recall()

    DESCRIPTION
        This function returns the recall of each class in the model

    RETURNS
        Returns the recall of the model
    """
    def recall(self):
        self.m_y_pred = self.m_model.predict(self.m_x_test_vec)
        return recall_score(self.m_y_test, self.m_y_pred, average=None)
    
    """
    Model::f1_score()

    NAME
        Model::f1_score() - Function that returns the f1 score of each class in the dataset
    
    SYNOPSIS
        list f1_score()

    DESCRIPTION
        This function returns the f1 score of each class in the model

    RETURNS
        Returns the f1 score of the model
    """
    def f1_score(self):
        self.m_y_pred = self.m_model.predict(self.m_x_test_vec)
        return f1_score(self.m_y_test, self.m_y_pred, average=None)
    
    """
    Model::hamming_loss()

    NAME
        Model::hamming_loss() - Function that returns the hamming loss of the model
    
    SYNOPSIS
        float hamming_loss()

    DESCRIPTION
        This function returns the hamming loss of the model

    RETURNS
        Returns the hamming loss of the model
    """
    def hamming_loss(self): 
        self.m_y_pred = self.m_model.predict(self.m_x_test_vec)
        return hamming_loss(self.m_y_test, self.m_y_pred)

    """
    Model::kfold()

    NAME
        Model::kfold() - Function that runs k-fold cross validation
                         where k = 5
    
    SYNOPSIS
        list kfold()

    DESCRIPTION
        This function runs k-fold cross validation where k = 5 and outputs the 
        Mean Squared Error for each fold.

    RETURNS
        Returns the the Mean Squared Error for each fold
    """
    def kfold(self):
        return (cross_val_score(self.m_model, self.m_x_vec, self.m_y, cv=5))

    """
    Model::confusion_matrix()

    NAME
        Model::confusion_matrix() - Function that returns the confusion matrix of the model
    
    SYNOPSIS
        list confusion_matrix()

    DESCRIPTION
        This function returns the confusion matrix of the model

    RETURNS 
        Returns the confusion matrix of the model
    """
    def confusion_matrix(self):
        self.m_y_pred = self.m_model.predict(self.m_x_test_vec)
        return confusion_matrix(self.m_y_test, self.m_y_pred)