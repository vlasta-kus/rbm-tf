#!/usr/bin/python

import os
import numpy as np
from pandas import DataFrame
from py2neo import Graph, Node, Relationship, authenticate

class NeoInterface:
    def __init__(self, stopwords_thr = 0.8):
        authenticate("localhost:7474", "neo4j", "neo")
        self.graph = Graph("http://localhost:7474/db/data/")

        self.stopwords_docFrequencyThreshold = stopwords_thr
        self.bowSize_sentences = 60
        self.bowSize_documents = 2000

        self.doSentences = True
        self.doDocuments = False

        self.stopwords = None
        self.BOW_global = None

        self.query_document_vectors = """
            match (n:{label})-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:HAS_TAG]->(t:Tag)
            with a, collect(distinct id(t)) as tags
            return id(a) as docId, extract(t in {bow} | (case when t in tags then 1 else 0 end)) as vector
        """
        self.query_single_document_vector = """
            match (n:{label})-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:HAS_TAG]->(t:Tag)
            where id(a) = {id}
            with a, collect(distinct id(t)) as tags
            return id(a) as docId, extract(t in {bow} | (case when t in tags then 1 else 0 end)) as vector
        """
       
        ### Queries for sentence-wise vectors 
        self.query_localBOW_and_sentence_vectors = """
            match (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[r:HAS_TAG]->(t:Tag)
            where id(a) = {idSpec} and not (id(t) in {stopwords}) and (ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) OR size(t.pos) = 0)
            with a, t, count(distinct s) as nSentences, sum(r.tf) as sumTf
            order by id(a), nSentences desc, sumTf desc
            with a, collect(id(t)) as tags
            with a, tags[..60] as BOW
            where size(BOW) = 60

            match (a)-[:CONTAINS_SENTENCE]->(s:Sentence)-[r:HAS_TAG]->(t:Tag)
            where not (id(t) in {stopwords}) and (ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) OR size(t.pos) = 0)
            with BOW, a, s, collect(distinct id(t)) as tags
            return id(s) as docId, extract(t in BOW | (case when t in tags then 1 else 0 end)) as vector
        """
        self.query_sentence_vectors = """
            // create batches
            match (n:{label})-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)
            //where not n.processed = true
            //with n, a
            //limit 50 // batch size!
            //set n.processed = true
            with n, id(a) as id
            
            // get BOW per document and sentence vectors
            %s
        """ % self.query_localBOW_and_sentence_vectors

    def encodeSentences(self, dimension=60):
        self.bowSize_sentences = dimension
        self.doSentences = True
        self.doDocuments = False
        
    def encodeDocuments(self, dimension = 2000):
        self.bowSize_documents = dimension
        self.doSentences = False
        self.doDocuments = True

    def getStopwords(self, label):
        query_stopwords = """
            match (n:{label})
            with count(*) as nDocs
            match (n:{label})-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(:Sentence)-[r:HAS_TAG]->(t:Tag)
            with t, 1.0 * count(distinct n) / nDocs as docFrequency
            where docFrequency > {thr}
            return collect(id(t)) as stoptagsIDs, collect(t.value) as stoptagsValues
        """
        result = self.graph.run(query_stopwords.format(label=label, thr = self.stopwords_docFrequencyThreshold)).data()[0]
        print(result['stoptagsValues'])
        return result['stoptagsIDs']

    def getGlobalBOW(self, label):
        # must be called _after_ stopwords were extracted
        # TO DO: decided BOW based on tf*idf?
        queryBOW_global = """
            match (:{label})-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:HAS_TAG]->(t:Tag)
            where ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]) OR size(t.pos) = 0
            with t, count(distinct a) as docCount
            where not (id(t) in {stopwords})
            with t, docCount
            order by docCount desc
            with collect(id(t)) as tags, collect(t.value) as tagValues
            return tags[..{dim}] as BOW, tagValues[..{dim}] as BOW_values
        """
        result = self.graph.run(queryBOW_global.format(stopwords=self.stopwords, dim=self.bowSize_documents, label=label)).data()[0]
        print(result['BOW_values'])
        return result['BOW']

    def trainDataFromNeo(self, label, outFile):
        file_name = self.getOutputFilename(outFile)
        if os.path.exists(file_name):
            print("\n > Output file %s already exists. Skipping data retrieval from Neo4j." % file_name)
            return

        print("\n Retrieving stopwords:")
        self.stopwords = self.getStopwords(label)

        print("\n > Retrieving document vectors")
        if self.doDocuments:
            print("\n Retrieving BOW:")
            self.BOW_global = self.getGlobalBOW(label)
            #print("   Saving to a file")
            #with open('BOW.out', 'w') as f:
            #    f.write(",".join(BOW))
            df = DataFrame(self.graph.run(self.query_document_vectors.format(label=label, bow=self.BOW_global)).data())
        elif self.doSentences:
            df = DataFrame(self.graph.run(self.query_sentence_vectors.format(label=label, stopwords=self.stopwords, idSpec="id")).data())

        print("   Saving to a file %s" % file_name)
        df.to_csv(file_name)
        
        #with graph.begin() as tx:
        #    cursor = tx.run(self.queryBOW)
        #    while cursor.forward():
        #        print(cursor.current['BOW'])

    def documentFromNeo(self, nodeId, label):
        # nodeId = id(AnnotatedText)
        if not self.stopwords:
            print("\n Retrieving stopwords:")
            self.stopwords = self.getStopwords(label)

        if self.doSentences:
            query = self.query_localBOW_and_sentence_vectors.format(idSpec=repr(nodeId), stopwords=self.stopwords)
        else:
            if not self.BOW_global:
                self.getGlobalBOW(label)
                print(" > Global BOW retrieved:")
                print(self.BOW_global)
            query = self.query_single_document_vector.format(id=nodeId, bow=self.BOW_global)

        df = DataFrame(self.graph.run(query).data())
        if df.empty:
            print(" ERROR: requested document does not contain enough unique tags")
            return None, None
        #return df.docId.values, self.vectorsOfStringsToNdarray(df.vector.values)
        return df.docId.values, np.vstack(df.vector.values)

    def getOutputFilename(self, outFile):
        if self.doSentences:
            app = "__sentences_" + repr(self.bowSize_sentences)
        else:
            app = "__documents_" + repr(self.bowSize_documents)
        file_name = outFile
        if outFile[-4:] == ".txt":
            file_name = file_name[:-4] + app + ".txt"
        return file_name

    def vectorsOfStringsToNdarray(self, ndarray):
        result = []
        for strvec in ndarray:
            try:
                result.append([float(j) for j in strvec[1:-1].split(",")])
            except:
                print(" Error! Couldn't transfrom this string to a list of floats: %s" % strvec)
                continue
        if len(result) == 0:
            return None
        return np.vstack(result)

    def vectorsToNeoNode(self, propertyKey, IDs, vectors):
        # `IDs` and `vectors` are numpy ndarrays
        query = """
        MATCH ({name}) WHERE id({name}) = {id}
        SET {name}.{key} = {vector}
        """

        finalQuery = ""
        name = ""
        for id, row in zip(IDs, vectors):
            if finalQuery != "":
                finalQuery += "\nWITH " + name
            name = "n"+repr(id)
            finalQuery += query.format(name=name, id=id, key=propertyKey, vector=row.tolist())
        self.graph.run(finalQuery)

