
from AEModel import AEModel
from NeoInterface import NeoInterface
from DataFromTxt import DataFromTxt


node_label = "TestText"
id__data_to_process = -1

# Neo4j handler
neo = NeoInterface()
neo.encodeSentences(60)
#neo.encodeDocuments(2000)

### Data for training
#file_with_train_data = "data/docVectors-NASA.out"
#file_with_train_data = "data/sentenceVectors-Emails-January.out"
file_with_train_data = "data/input_train_data.txt"
neo.trainDataFromNeo(node_label, file_with_train_data)

### Initialize the main model
model = AEModel()

### Training
model.train(file_with_train_data, save=True) # train & save

### Predict
#IDs, data = documentFromNeo(id__data_to_process, node_label) # retrieve data as a numpy ndarray
#prediction = model.predict(data) # load model & predict
#neo.vectorsToNeoNode('nn_encoded_vector', IDs, data)
