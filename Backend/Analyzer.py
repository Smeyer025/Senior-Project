##############################################################
# Analyzer Class - Act as the liaison between React Frontend #
#                  and Flask/python backend                  #
##############################################################

import Model

"""
Analyzer

NAME
    Analyzer: Analyzer class; acts as liaison between frontend and Model class

SYNOPSIS
    __init__(): Constructor
    setModel(): Set new model type, create new model
    setDataset(): Set new dataset choice, create new model
    predict(): Use model to make a prediction 

DESCRIPTION
    This class provides functionality that controls the Model class and ensures
    the frontend can access it easily. 
"""
class Analyzer:
    DATASETS = ["SocialMedia", "AirlineReviews", "DrugReviews", "HotelReviews", "MovieReviews"]
    MODEL_TYPES = ["LogisticRegression", "SupportVectorMachine", "RandomForest", "KNearestNeighbors", "VotingClassifier"]
    importedDatasetColumns = {}
    m_currModel = ""
    m_currDataset = ""
    m_currModelType = ""
    m_currTextCol = ""
    m_currSentCol = ""
    m_currPosLabel = ""
    m_currNegLabel = ""
    m_currNeutLabel = ""
    m_currBalance = False

    # Static Analyzer object
    ana = "ana not initialized yet"

    """
    Analyzer::__init__()

    NAME
        Analyzer::__init__() - Constructor for Analyzer Class
    
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
        This constructor takes in all necessary variables for analysis, ensures the dataset and model choices are valid,
        assigns all member variables, and creates the Model based on the parameters given.
    """
    def __init__(self, a_datasetChoice, a_modelType, a_textColumn, a_sentimentColumn, a_posLabel="positive", a_negLabel="negative", a_neutLabel="neutral", a_balance = False):
        if a_datasetChoice not in self.DATASETS:
            raise Exception("Invalid dataset chosen")
        if a_modelType not in self.MODEL_TYPES:
            raise Exception("Invalid model type chosen")
        
        self.m_currDataset = a_datasetChoice
        self.m_currModelType = a_modelType
        self.m_currTextCol = a_textColumn
        self.m_currSentCol = a_sentimentColumn
        self.m_currPosLabel = a_posLabel
        self.m_currNegLabel = a_negLabel
        self.m_currNeutLabel = a_neutLabel
        self.m_currBalance = a_balance
        self.m_currModel = Model.Model(self.m_currDataset, self.m_currModelType, self.m_currTextCol, self.m_currSentCol, self.m_currPosLabel, self.m_currNegLabel, self.m_currNeutLabel, self.m_currBalance)
        
    """
    Model::setModel()

    NAME
        Analyzer::setModel() - Function that changes the model in the Analyzer object 
    
    SYNOPSIS
        void setModel(self, a_modelType)
            self        --> Required first parameter for any Python class function/constructor. The object being created/operated on 
            a_modelType --> Updated model type

    DESCRIPTION
        This function takes in the type of model to be changed to, ensures it's in the list of acceptable models,
        updates the current model type in the Analyzer object, and generates a new model.
    """
    def setModel(self, a_modelType):
        if a_modelType not in self.MODEL_TYPES:
            raise Exception("Invalid model type chosen")
        
        self.m_currModelType = a_modelType 
        self.m_currModel = Model.Model(self.m_currDataset, self.m_currModelType, self.m_currTextCol, self.m_currSentCol, self.m_currPosLabel, self.m_currNegLabel, self.m_currNeutLabel, self.m_currBalance)

    """
    Model::setDataset()

    NAME
        Analyzer::setDataset() - Function that changes the datasets in the Analyzer object 
    
    SYNOPSIS
        void setDataset(self, a_datasetChoice, a_posLabel="positive", a_negLabel="negative", a_neutLabel="neutral", a_balance = False)
            self        --> Required first parameter for any Python class function/constructor. The object being created/operated on 
            a_datasetChoice   --> Dataset chosen for the model
            a_posLabel        --> Label denoting positive tone
            a_negLabel        --> Label denoting negative tone
            a_neutLabel       --> Label denoting neutral tone
            a_balance         --> Boolean containing whether or not the dataset should be balanced as part of data cleaning

    DESCRIPTION
        This function takes in the type of dataset to be changed to, the positive, negative, and neutral labels, 
        and the boolean containing whether or not to balance the data, makes sure the dataset choice is 
        valid, sets all the necessary member variables, and generates a new model based on the updated dataset.
    """
    def setDataset(self, a_datasetChoice, a_posLabel="positive", a_negLabel="negative", a_neutLabel="neutral", a_balance = False):
        if a_datasetChoice not in self.DATASETS:
            raise Exception("Invalid dataset chosen")
        
        self.m_currDataset = a_datasetChoice
        self.m_currPosLabel = a_posLabel
        self.m_currNegLabel = a_negLabel
        self.m_currNeutLabel = a_neutLabel
        self.m_currBalance = a_balance
        self.m_currModel = Model.Model(self.m_currDataset, self.m_currModelType, self.m_currTextCol, self.m_currSentCol, self.m_currPosLabel, self.m_currNegLabel, self.m_currNeutLabel, self.m_currBalance)

    """
    Model::predict()

    NAME
        Analyzer::predict() - Function that takes in text to be predicted and returns the prediction
    
    SYNOPSIS
        String predict(self, a_textToPredict)
            self        --> Required first parameter for any Python class function/constructor. The object being created/operated on 
            a_modelType --> Updated model type

    DESCRIPTION
        This function takes in the type of model to be changed to, ensures it's in the list of acceptable models,
        updates the current model type in the Analyzer object, and generates a new model.

    RETURNS
        String containing the prediction the model made about the tone of the text sample (positive, neutral, negative)
    """
    def predict(self, a_textToPredict):
        return self.m_currModel.predict(a_textToPredict)
