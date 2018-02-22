
from AEModel import AEModel
from NeoInterface import NeoInterface
from DataFromTxt import DataFromTxt


node_label = "Document" #"TestText"
id__data_to_process = 93848 #219132 # AnnotatedText

# Neo4j handler
neo = NeoInterface(stopwords_thr = 0.04)
#neo.encodeSentences(50)
neo.encodeDocuments(2000)

### Data for training
#file_with_train_data = "data/docVectors-NASA.out"
#file_with_train_data = "data/sentenceVectors-Emails-January.out"
file_with_train_data = "data/input_train_data.txt"
neo.trainDataFromNeo(node_label, file_with_train_data)

### Initialize the main model
model = AEModel()

### Training
model.train(file_with_train_data, train_fraction=0.9, vector_nonzero_fraction=0.006, save=True) # for documents
#model.train(file_with_train_data, train_fraction=0.9, vector_nonzero_fraction=0.1, save=True) # for sentences

### Predict
#IDs, data = neo.documentFromNeo(id__data_to_process, node_label) # retrieve data as a numpy ndarray
#if data.size > 0:
#    predictions = model.predict(data) # load model & predict
#    neo.vectorsToNeoNode('nn_encoded_vector', IDs, predictions)
