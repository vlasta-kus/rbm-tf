
from AEModel import AEModel
from NeoInterface import NeoInterface
from DataFromTxt import DataFromTxt


node_label = "Lesson" #"TestText"
nonzero_frac = 0.01 # specify minimal fraction of non-zero vector elements

# Neo4j handler
neo = NeoInterface(stopwords_thr = 0.8)
neo.encodeSentences(800)
#neo.encodeDocuments(2000)

### Data for training: from a file
#file_with_train_data = "data/docVectors-NASA-1000.out"
file_with_train_data = "data/sentenceGlobalVectors-NASA-800.out"
#file_with_train_data = "data/sentenceVectors-Emails-January.out"
#file_with_train_data = "data/input_train_data__Chris_sentences_50.txt"
#file_with_train_data = "data/input_train_data.txt"

### Data for training: from Neo4j
neo.trainDataFromNeo(node_label, file_with_train_data)

### Initialize the main model
model = AEModel()

### Training
#model.train(file_with_train_data, train_fraction=0.9, vector_nonzero_fraction=nonzero_frac, save=True) # for documents
#model.train(file_with_train_data, train_fraction=0.9, vector_nonzero_fraction=nonzero_frac, save=True) # for sentences

### Predict: full training set
IDs, vectors = model.predictDataset(file_with_train_data, nonzero_frac)
neo.vectorsToNeoNode('ae_vec', IDs, vectors)

### Predict: single document
id__data_to_process = 93848 #219132 # AnnotatedText
#IDs, data = neo.documentFromNeo(id__data_to_process, node_label) # retrieve data as a numpy ndarray
#if data.size > 0:
#    predictions = model.predict(data) # load model & predict
#    neo.vectorsToNeoNode('nn_encoded_vector', IDs, predictions)
