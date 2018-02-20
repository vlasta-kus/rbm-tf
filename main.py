
from AEModel import AEModel
from DataFromNeo import trainingDataFromNeo, documentFromNeo
from DataFromTxt import DataFromTxt


label__train_data = ""
id__data_to_process = -1


# Data for training
file_with_train_data = "data/input_train_data.txt"
trainingDataFromNeo(label__train_data, file_with_train_data)

# Data already prepared (bypass `trainingDataFromNeo`)
#file_with_train_data = "data/docVectors-NASA.out"
#file_with_train_data = "data/sentenceVectors-Emails-January.out"

# Initialize the main model
model = AEModel()

# Training
model.train(file_with_train_data, save=True) # train & save

# Predict
#data = documentFromNeo(id__data_to_process) # retrieve data as a numpy ndarray
#prediction = model.predict(data) # load model & predict
#saveToNeo(prediction)

