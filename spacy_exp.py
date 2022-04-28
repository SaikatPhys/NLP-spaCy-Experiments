HEAD = '\033[95m'#magenta
OKB = '\033[94m'#blue
OKG = '\033[92m'#green
WARN = '\033[93m'#yellow ochre
FAIL = '\033[91m'#red
ENDC = '\033[0m'#ends an effect. So, use like HEAD+ULINE+"Hello"+ENDC+ENDC
BOLD = '\033[1m'
ULINE = '\033[4m'

import numpy as np
import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt

try:
	import spacy
except ImportError :
	print (FAIL+BOLD+"NLP package Spacy is missing. Install spacy using the instructions here: https://spacy.io/usage"+ENDC+ENDC)
	raise ImportError ( "No module named Spacy" )

try:
	nlp = spacy.load("en_core_web_sm")
except  IOError:
	print (FAIL+BOLD+"English model in Spacy is missing. You can download the -en model: python -m spacy download en"+ENDC+ENDC)
	raise  IOError ( "No English language model available within Spacy" )

try:
	from spacy import displacy
except ImportError :
	print (FAIL+BOLD+"DisplaCy is missing. This is availble in Spacy v2.0. Update Spacy."+ENDC+ENDC)

try:
	from pathlib import Path
except ImportError :
	raise ImportError ( "Path not found from pathlib" )

try:
	from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
except ImportError :
	raise ImportError ("No Count vectorizer and TF-IDF vectorizer found in scikit-learn feature extraction module. Install scikit-learn")

#from sklearn.base import TransformerMixin
#from sklearn.pipeline import Pipeline

#--------------------------------------------------------------------------------

# TASK 1

#Tokenization: segmenting a text document into words, that are
#also known as tokens. A token is defined as a sequence of characters
#that together forms a semantic unit while processing of the text.

def token_attributes(text):
	doc = nlp(text)
	print(OKG+BOLD+"Tokenization: segmenting a given text into words using Natural Language Processing (NLP)."+ENDC+ENDC)
	print(OKG+BOLD+"A token is defined as a sequence of characters that together forms a semantic unit while processing of the text."+ENDC+ENDC)
	print(OKG+BOLD+"In the following you also see Lexicon Normalization by means of lemmatization -- a first step in the text data cleaning process that converts high dimensional features into low dimensional features."+ENDC+ENDC)
	for token in doc:
		print("Token: %s - Lemma: %s, Shape: %s, Stop Word: %s, Punctuation: %s, URL: %s, Email: %s, Number: %s, "
			  "Entity Type: %s" % (FAIL+token.text+ENDC, token.lemma_, token.shape_, token.is_stop, token.is_punct, token.like_url, token.like_email, token.like_num, token.ent_type_))
		print("---------------------")

#--------------------------------------------------------------------------------

# TASK 2

# sentence splitting / sentence boundary disambiguation,
# sentence segmentation / sentence boundary detection.

def split_into_sentences(input_text):
	print(OKG+BOLD+"Split a given text into complete sentences using Natural Language Processing (NLP)."+ENDC+ENDC)
	print("---------------------")
	for i, sentence in enumerate(nlp(input_text).sents):
		print("Sentence %d: %s" % (i, WARN+sentence.text+ENDC))
		print("Start Offset %d, End Offset %d" % (sentence.start, sentence.end))
		print("---------------------")

#--------------------------------------------------------------------------------

# TASK 3

#Part-of-speech (POS) tagging: the process of tagging a word
#with its corresponding part-of-speech like noun, adjective,
#verb, adverb, etc.

def pos_tagging(text):
	print(OKG+BOLD+"Part-of-speech (POS) tagging using Natural Language Processing (NLP)."+ENDC+ENDC)
	print(OKG+BOLD+"This process tags words in a given text with its corresponding part-of-speech like noun, adjective, verb, adverb, etc."+ENDC+ENDC)
	print("---------------------")
	doc = nlp(text)
	for token in doc:
		print("Token: %s, Coarse-grained POS-Tag: %s" % (FAIL+token.text+ENDC, WARN+token.pos_+ENDC))
		print("Token: %s, Fine-grained POS-Tag: %s" % (token.text, token.tag_))
		print("---------------------")

#--------------------------------------------------------------------------------

# TASK 4

# Named Entity Recognition (NER) is the process of tagging sequences
# of words in a given piece of text as a person, organization, place etc.
#  It is a fundamental task in NLP and useful in text
# classification, search and indexing, recommendation,
# keyword extraction, knowledge graphs and so on.

# The following function also creates a visual color map of
# various categories of entities. It shows in a local-host at http server.


def extract_entities(text):
	print(OKG+BOLD+"Named Entity Recognition (NER) or Entity extraction using Natural Language Processing (NLP)."+ENDC+ENDC)
	print(OKG+BOLD+"Process of NER tags sequences of words in a given text as a person, org., place, etc."+ENDC+ENDC)
	print("---------------------")
	for entity in nlp(text).ents:
		print("Entity: ", FAIL+entity.text+ENDC)
		print("Entity Type: %s | %s" % (entity.label_, spacy.explain(entity.label_)))
		print("Label of the named Entity: ", entity.label)
		print("Start Offset of the Entity: ", entity.start_char)
		print("End Offset of the Entity: ", entity.end_char)
		print("---------------------")
	displacy.serve(nlp(text), style = "ent")
	#output_path = Path("/Users/saikat/factsandfakes/experiments/Named_Entity_Recognition.svg")
	#output_path.open("w").write(NER)

#--------------------------------------------------------------------------------

# TASK 5

# Noun Phrase Chunking: Phrase chunking is the dependency parsing
# process of dividing sentences into non-overlapping phrases.


def noun_phrases(text):
	print(OKG+BOLD+"Noun Phrase Chunking: Phrase chunking is the process of dividing sentences into non-overlapping phrases using Natural Language Processing (NLP)."+ENDC+ENDC)
	print("---------------------")
	for np in nlp(text).noun_chunks:
		print("Noun Phrase: ", FAIL+np.text+ENDC)
		print("Root of the Noun Phrase: ", np.root.text)
		print("Dependency of the Noun Phrase: ", np.root.dep_)
		print("Head text of the Noun Phrase: ", np.root.head.text)
		print("Start Offset of the Noun Phrase: ", np.start_char)
		print("End Offset of the Noun Phrase: ", np.end_char)
		print("---------------------")

#--------------------------------------------------------------------------------

# TASK 6

# Dependency diagram or syntactic dependency parsing
# Creates dependency diagrams of each sentences and saves as svg files.
# see https://spacy.io/api/annotation#dependency-parsing for
# interpretations of the Universal Dependency Labels.

def visual_dependency_parse(text):
	print(OKG+BOLD+"Syntactic Dependency Parsing: Creates dependency diagrams of each individual sentences and saves in svg files"+ENDC+ENDC)
	sentence_spans = list(nlp(text).sents)
	for i in range (len(sentence_spans)):
		svg = displacy.render(sentence_spans[i], style="dep")
		output_path = Path("/Users/saikat/factsandfakes/experiments/sentence_" + str(i) + ".svg")
		output_path.open("w", encoding="utf-8").write(svg)


#--------------------------------------------------------------------------------

# TASK 7

#Text Similarity: In order to calculate similarity between
# two text snippets, the usual way is to convert the text
# into its corresponding vector representation, for which
# there are many methods like one-hot encoding of text, and
# then calculate similarity or difference using different
# distance metrics such as "cosine-similarity" and "euclidean
# distance" applicable to vectors.

def text_similarity(text1, text2):
	doc1 = nlp(text1)
	doc2 = nlp(text2)
	print(OKG+BOLD+"Cosine similarity between the two documents:"+ENDC+ENDC)
	print (doc1.similarity(doc2))


def token_similarity(doc):
	for token1 in doc:
		for token2 in doc:
			print("Token 1: %s, Token 2: %s - Similarity: %f" % (token1.text, token2.text, token1.similarity(token2)))
			print("---")

#--------------------------------------------------------------------------------

# TASK 8

# Removing the stop words, punctuations from a given text,
# And coverting each token to lower case.

def rm_stop_words (text):
	#print(OKG+BOLD+"This function filters the stop-words (noise) from the text"+ENDC+ENDC)
	#print("---------------------")
	spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS #list of STOP WORDS
	punctuations = string.punctuation # list of punctuations
	doc = nlp(text)

	"""
	filtered_sent=[]
	for word in doc:
		if (word.is_stop==False)  and (word.is_punct==False):
			filtered_sent.append(word)
	filtered_sentence = " ".join(str(x) for x in filtered_sent)
	"""

	filtered_sent = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
	filtered_sent = [word for word in filtered_sent if word not in spacy_stopwords and word not in punctuations]
	#filtered_sentence = " ".join(str(x) for x in filtered_sent)

	#print(FAIL+"Filtered sentence after removing the stop words, punctuations and made lower-case :"+ENDC)
	print (len(filtered_sent))
	print (filtered_sent)
	print("---------------------")
	return filtered_sent

#--------------------------------------------------------------------------------

# TASK 9

# Bag of Words (BoW): it converts text into the matrix of
# occurrence of words within a given document. It focuses on
# whether given words occurred or not in the document, and it
# generates a matrix that we might see referred to as
# a BoW matrix or a document term matrix.

def count_vectorizer (text):
	# preprocess the data
	print(FAIL+"Filtered training document after removing the stop words, punctuations and made lower-case :"+ENDC)
	text = rm_stop_words (text)

	# create the transform
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_df=1.0, min_df=0.02, max_features=None)

	# tokenize and learn the vocabularies from one or two documents.
	vectorizer.fit(text)

	# summarize
	print(OKG+BOLD+"Summarizing the vocabulary:"+ENDC+ENDC)
	print (len(vectorizer.vocabulary_))
	print(vectorizer.vocabulary_)
	print(vectorizer.get_feature_names())


	# encode the document. An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.
	vector = vectorizer.transform(text)

	print("---------------------")

	print(OKG+BOLD+"Summarize the encoded vector"+ENDC+ENDC)
	# summarize encoded vector
	print(vector.shape)
	print(vector.toarray())

	print("---------------------")



#--------------------------------------------------------------------------------

# TASK 10


# TF-IDF (Term Frequency-Inverse Document Frequency):
# a way of normalizing our Bag of Words(BoW) by looking
# at each word’s frequency in comparison to the document
# frequency. In other words, it’s a way of representing
# how important a particular term is in the context of a
# given document, based on how many times the term appears
# and how many other documents that same term appears in.
# The higher the TF-IDF, the more important that term is to that document.

def TFIDF (document, text):
	vectorizer = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')

	#------------ training & IDF ---------------
	print(HEAD+BOLD+"Computing Inverse Document Frequency (IDF)"+ENDC+ENDC)
	print(FAIL+"Filtered training document after removing the stop words, punctuations and made lower-case :"+ENDC)
	filtered_sent = rm_stop_words (document)

	# tokenize and learn the vocabularies from one or two documents.
	vectorizer.fit(filtered_sent)

	print(OKG+BOLD+"Summarizing the vocabulary:"+ENDC+ENDC)
	print (len(vectorizer.vocabulary_))
	print (vectorizer.vocabulary_)
	print("---------------------")

	print(OKG+BOLD+"IDF of the vocabularies"+ENDC+ENDC)
	print(dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)))
	print("---------------------")

	feature_names = np.array(vectorizer.get_feature_names())
	sorted_by_idf = np.argsort(vectorizer.idf_)

	print(OKG+BOLD+"Features with lowest idf:"+ENDC+ENDC)
	print(feature_names[sorted_by_idf[:10]])
	print(OKG+BOLD+"Features with highest idf:"+ENDC+ENDC)
	print(feature_names[sorted_by_idf[-10:]])
	print("---------------------")

	#------ plotting inverse document frequency -------
	rr = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

	token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()
	token_weight.columns=('token','weight')
	token_weight = token_weight.sort_values(by='weight', ascending=False)

	sns.barplot(x='token', y='weight', data=token_weight)
	plt.title("Inverse Document Frequency(IDF) per token")
	fig=plt.gcf()
	fig.set_size_inches(10,5)
	plt.show()

	#---------------------------------------------------

	#------- testing & computing TF-IDF --------
	print(HEAD+BOLD+"Computing Term Frequency Inverse Document Frequency (TF-IDF)"+ENDC+ENDC)
	print(FAIL+"Filtered test document after removing the stop words, punctuations and made lower-case :"+ENDC)
	filtered_text = rm_stop_words (text)

	# encode a text after removing stop-words and punctuations
	vector = vectorizer.transform(filtered_text)

	# find maximum value for each of the features over all of dataset:
	max_val = vector.max(axis=0).toarray().ravel()

	print(OKG+BOLD+"TF-IDF of the vocabularies"+ENDC+ENDC)
	print(dict(zip(vectorizer.get_feature_names(), max_val)))
	print("---------------------")


	#sort weights from smallest to biggest and extract their indices
	sort_by_tfidf = max_val.argsort()

	print(OKG+BOLD+"Features with lowest tfidf:"+ENDC+ENDC)
	print(feature_names[sort_by_tfidf[:10]])
	print(OKG+BOLD+"Features with highest tfidf:"+ENDC+ENDC)
	print(feature_names[sort_by_tfidf[-10:]])
	print("---------------------")

	#------ plotting term frequency inverse document frequency -------

	rr1 = dict(zip(vectorizer.get_feature_names(), max_val))

	token_weight = pd.DataFrame.from_dict(rr1, orient='index').reset_index()

	token_weight.columns=('token','weight')
	token_weight = token_weight.sort_values(by='weight', ascending=False)

	sns.barplot(x='token', y='weight', data=token_weight)
	plt.title("Term Frequency-Inverse Document Frequency(TF-IDF) per token")
	fig=plt.gcf()
	fig.set_size_inches(10,5)
	plt.show()


#--------------------------------------------------------------------------------

if __name__ == "__main__" :
	test_text = open("exmaple1.txt").read()
	text = open("example2.txt").read()
	
	#token_attributes (text)
	#split_into_sentences (text)
	#pos_tagging (text)
	#extract_entities(text)
	#noun_phrases(text)
	#visual_dependency_parse(text)
	#count_vectorizer (text)


	#text_similarity (text, test_text)
	TFIDF (text, test_text)
