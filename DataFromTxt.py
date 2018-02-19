import sys
import pandas as pd
import numpy as np

class DataFromTxt:
    """ Read train/test data from .txt file """

    def __init__(self, path, nonzeros_frac=0.1):
        self.path = path
        self.min_nonzero_frac = nonzeros_frac
        self.inputDf = pd.DataFrame({'lineIdx': [], 'vector': []})
        self.dimension = 0

        # process the input file
        print(" Processing input file ...")
        with open(path, 'r') as f:
            idx = -1
            for line in f:
                idx += 1
                if idx == 0: # ignore head (column names)
                    continue
                vec = self.processLine(line, idx)
                if vec:
                    self.inputDf = self.inputDf.append({'lineIdx': idx, 'vector': np.array(vec)}, ignore_index=True)

        print(" Input DataFrame size is %d." % self.inputDf.lineIdx.count())

        # get prevailing vector dimension
        self.inputDf['dim'] = self.inputDf['vector'].apply(lambda vec: len(vec))
        self.dimension = np.argmax(self.inputDf.groupby('dim').lineIdx.count())
        print(" Prevailing vector dimension: %d" % self.dimension)
        
        # final cleaning: 
        #       * make sure only documents of the prevailing dimension are used
        #       * only documents that contain some minimal number of non-null elements
        self.inputDf['n_nonzero'] = self.inputDf['vector'].apply(lambda vec: np.count_nonzero(vec))
        self.finalInput = self.inputDf.query('dim == %d & n_nonzero >= %d' % (self.dimension, self.min_nonzero_frac * self.dimension)) #[self.inputDf.dim == self.dimension]
        self.n_documents = self.finalInput.lineIdx.count()
        print(" n_documents: %d" % self.n_documents)

        # placeholders for train and test sets
        self.finalInput_train = pd.DataFrame()
        self.finalInput_test = pd.DataFrame()
        self.n_documents_train = 0
        self.n_documents_test = 0

    def splitTrainTest(self, frac=0.8):
        print(" Splitting to train and test ...")
        self.finalInput_train = self.finalInput.sample(frac=frac, random_state=123)
        self.finalInput_test = self.finalInput.drop(self.finalInput_train.index)
        self.n_documents_train = self.finalInput_train.lineIdx.count()
        self.n_documents_test = self.finalInput_test.lineIdx.count()
        print(" Train dataset size: %d" % self.n_documents_train)
        print(" Test dataset size: %d" % self.n_documents_test)

    def getTrainData(self):
        if not self.finalInput_train.empty:
        #    return self.dfToNumpy(self.finalInput_train)
            return DataIterator(self.finalInput_train)
        print(" ERROR: train dataset does not exist!")
        return None
        #return self.dfToNumpy(self.finalInput)

    def getTestData(self):
        if not self.finalInput_test.empty:
        #    return self.dfToNumpy(self.finalInput_test)
            return DataIterator(self.finalInput_test)
        print(" ERROR: test dataset does not exist!")
        return None

    def dfToNumpy(self, df):
        return np.vstack(df.vector.values)

    def processLine(self, line, idx):
        if "[" not in line:
            print(" WARN: skipping line %d because it does not contain required delimiter")
            return None
        strVec = line.rstrip().split("[")[1][:-2]
        try:
            vec = [float(i) for i in strVec.split(",")]
        except Exception as e:
            print(" ERROR transforming input at line %d to a vector: cannot convert to float" % idx)
            return None
        return vec


class DataIterator:
    """ Simple custom data iterator """
    def __init__(self, df):
        self.df = df
        self.n_documents = df.lineIdx.count()
        self.current_position = 0
        self._epochs_completed = 0

    def next_batch(self, batch_size):
        if not self.has_next():
            self.reset()
        end = self.current_position + batch_size

        if not self.df.empty:
            if end > self.n_documents:
                end = self.n_documents
                self._epochs_completed += 1
            df = self.df[self.current_position : end]
        else:
            if end > self.n_documents:
                end = self.n_documents
                self._epochs_completed += 1
            df = self.finalInput[self.current_position : end]

        self.current_position = end
        return self.dfToNumpy(df), None

    def has_next(self):
        if not self.df.empty:
            return self.current_position < self.n_documents
        return self.current_position < self.n_documents

    def reset(self):
        self.current_position = 0

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def getNumpyRepresentation(self):
        return self.dfToNumpy(self.df)

    def dfToNumpy(self, df):
        return np.vstack(df.vector.values)



if __name__ == "__main__":
    assert len(sys.argv) > 1, " ERROR: missing command-line argument (path to a data file)"
    data = DataFromTxt(sys.argv[1])
