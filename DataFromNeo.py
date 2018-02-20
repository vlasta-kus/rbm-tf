#!/usr/bin/python

from pandas import DataFrame
from py2neo import Graph, Node, Relationship, authenticate
from bs4 import BeautifulSoup
from bs4.element import Comment
import os

def dataFromNeo(label, outFile):
    authenticate("localhost:7474", "neo4j", "neo")
    graph = Graph("http://localhost:7474/db/data/")
    
    queryBOW = """
    match (a:AnnotatedText)
    with count(*) as nDocs
    match (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:HAS_TAG]->(t:Tag)
    where ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]) OR size(t.pos) = 0
    with t.id as tagId, count(distinct a) as docCount, 1.0 * count(distinct a)/nDocs as docFrequency
    where docFrequency < 0.8
    with tagId, docCount
    order by docCount desc
    with collect(tagId) as tags
    return tags[..2000] as BOW
    """
    
    queryDocTags = """
    match (a:AnnotatedText)
    with count(*) as nDocs
    match (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:HAS_TAG]->(t:Tag)
    where ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"])
    with t.id as tagId, count(distinct a) as docCount, 1.0 * count(distinct a)/nDocs as docFrequency
    where docFrequency < 0.9
    with tagId, docCount
    order by docCount desc
    with collect(tagId) as tags
    with tags[..2000] as BOW
    
    match (a:AnnotatedText)-[:CONTAINS_SENTENCE]->(s:Sentence)-[:HAS_TAG]->(t:Tag)
    with BOW, a, collect(distinct t.id) as tags
    return id(a) as docId, extract(t in BOW | (case when t in tags then 1 else 0 end)) as vector
    """
    
    # This query is specific for the database design of test e-mails for summarization
    #   * BOW is constructed per document and only from tags that occur in < 90 % of documents
    #   * BOW is constructed from top 60 tags in given document, tags are sorted in this order of importance:
    #       - number of sentences they occur in
    #       - total number of occurrences in given document (i.e. term-frequency tf)
    #       - TO DO: tf * idf values (they must be already calculated and stored in HAS_TAG relationships)
    queryDocTags_perDocument = """
    // get stopwords
    match (n:TestEmail)
    with count(*) as nDocs
    match (t:Tag)<-[*4..4]-(n:TestEmail)
    with t, 1.0 * count(distinct n) / nDocs as docFrequency
    where docFrequency > 0.4 //and (ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]) OR size(t.pos) = 0)
    with collect(id(t)) as stoptagsIDs
    
    // create batches
    match (n:TestEmail)
    //where not n.processed = true
    //with n, stoptagsIDs
    //limit 50 // batch size!
    //set n.processed = true
    with n, stoptagsIDs
    
    // get BOW per document
    match (n)-[:CONTAINS_SENTENCE]->(s:TestSentence)-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(:Sentence)-[r:HAS_TAG]->(t:Tag)
    where not (id(t) in stoptagsIDs) and (ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) OR size(t.pos) = 0)
    with n, t.value as tagVal, count(distinct s) as nSentences, sum(r.tf) as sumTf, stoptagsIDs
    order by id(n), nSentences desc, sumTf desc
    with n, collect(tagVal) as tags, stoptagsIDs
    with n, tags[..60] as BOW, stoptagsIDs
    where size(BOW) = 60
    
    // create sentence vectors
    match (n)-[:CONTAINS_SENTENCE]->(s:TestSentence)-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(:Sentence)-[r:HAS_TAG]->(t:Tag)
    where not (id(t) in stoptagsIDs) and (ANY(pos IN t.pos WHERE pos IN ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) OR size(t.pos) = 0)
    with BOW, n, s, collect(distinct t.value) as tags
    return id(s) as docId, extract(t in BOW | (case when t in tags then 1 else 0 end)) as vector
    """
    
    
    #df = DataFrame(graph.run(queryBOW).data())
    #print(df)
    
    #print("\n > Retrieving BOW")
    #BOW = graph.run(queryBOW).data()[0]['BOW']
    #print("   Saving to a file")
    #with open('BOW.out', 'w') as f:
    #    f.write(",".join(BOW))
    
    print("\n > Retrieving document vectors")
    #df = DataFrame(graph.run(queryDocTags).data()) # documents
    df = DataFrame(graph.run(queryDocTags_perDocument).data()) # sentences
    print("   Saving to a file")
    df.to_csv(outFile)
    
    
    #with graph.begin() as tx:
    #    cursor = tx.run(queryBOW)
    #    while cursor.forward():
    #        print(cursor.current['BOW'])
