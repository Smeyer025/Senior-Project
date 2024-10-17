import Model

class Analyzer:
    DATASETS = ["SocialMedia", "AirlineReviews", "DrugReviews", "HotelReviews", "MovieReviews"]
    MODEL_TYPES = ["LogisticRegression", "SupportVectorMachine", "RandomForest", "KMeansClustering"]
    currModel = ""
    currDataset = ""
    currModelType = ""
    currTextCol = ""
    currSentCol = ""
    currPosLabel = ""
    currNegLabel = ""
    currNeutLabel = ""

    def __init__(self, a_datasetChoice, a_modelType, a_textColumn, a_sentimentColumn, a_posLabel="positive", a_negLabel="negative", a_neutLabel="neutral"):
        if a_datasetChoice not in self.DATASETS:
            raise Exception("Invalid dataset chosen")
        if a_modelType not in self.MODEL_TYPES:
            raise Exception("Invalid model type chosen")
        
        self.currDataset = a_datasetChoice
        self.currModelType = a_modelType
        self.currTextCol = a_textColumn
        self.currSentCol = a_sentimentColumn
        self.currPosLabel = a_posLabel
        self.currNegLabel = a_negLabel
        self.currNeutLabel = a_neutLabel

        self.currModel = Model.Model(self.currDataset, self.currModelType, self.currTextCol, self.currSentCol, self.currPosLabel, self.currNegLabel, self.currNeutLabel)
        
    def setModel(self, a_modelType):
        if a_modelType not in self.MODEL_TYPES:
            raise Exception("Invalid model type chosen")
        
        self.currModelType = a_modelType 
        self.currModel = Model.Model(self.currDataset, self.currModelType, self.currTextCol, self.currSentCol, self.currPosLabel, self.currNegLabel, self.currNeutLabel)

    def setDataset(self, a_datasetChoice, a_posLabel="positive", a_negLabel="negative", a_neutLabel="neutral"):
        if a_datasetChoice not in self.DATASETS:
            raise Exception("Invalid dataset chosen")
        
        self.currDataset = a_datasetChoice
        self.currPosLabel = a_posLabel
        self.currNegLabel = a_negLabel
        self.currNeutLabel = a_neutLabel
        self.currModel = Model.Model(self.currDataset, self.currModelType, self.currTextCol, self.currSentCol, self.currPosLabel, self.currNegLabel, self.currNeutLabel)

    def predict(self, a_textToPredict):
        return self.currModel.predict(a_textToPredict)

# a = Analyzer("SocialMedia", "LogisticRegression", "clean_text", "category")
# print(a.predict("I love this so much!"))