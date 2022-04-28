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




#--------------------------------------------------------------------------------

# TASK 1

# Removing the stop words, punctuations from a given text,
# And coverting each token to its lemma (& lower case: optional)

def rm_stop_words (text):
	#print(OKG+BOLD+"This function filters the stop-words (noise) from the text"+ENDC+ENDC)
	#print("---------------------")
	spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS #list of STOP WORDS
	punctuations = string.punctuation # list of punctuations
	doc = nlp(text)


	#filtered_sent = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
	filtered_sent = [word.lemma_ for word in doc]
	filtered_sent = [word for word in filtered_sent if word not in spacy_stopwords and word not in punctuations]

	filtered_sentence = " ".join(str(x) for x in filtered_sent)

	#print(FAIL+"Filtered sentence after removing the stop words, punctuations and made lower-case :"+ENDC)
	print (len(filtered_sent))
	print (filtered_sent)
	print("---------------------")
	return filtered_sent




#--------------------------------------------------------------------------------

# TASK 2

# Text Similarity: In order to calculate similarity between
# two text snippets, the usual way is to convert the text
# into its corresponding vector representation, for which
# there are many methods like one-hot encoding of text, and
# then calculate similarity or difference using different
# distance metrics such as "cosine-similarity" and "euclidean
# distance" applicable to vectors.


def token_similarity(text):

	text = rm_stop_words (text)
	text = " ".join(str(x) for x in text)

	doc = nlp(text)


	"""
	for token1 in doc:
		for token2 in doc:
			print((token1.text, token2.text), "similarity =>", token1.similarity(token2))
			print("---")
	"""

	docx_similar = np.array([token1.similarity(token2) for token2 in doc for token1 in doc]).reshape(len(doc), -1)


	#df = pd.DataFrame(docx_similar)
	#df.columns = ["Token1","Token2","Similarity"]
	#print (df)


	plt.figure()
	plt.title("Cosine-similarities between the words")
	sns.heatmap(docx_similar, annot=True, cmap="YlGnBu")
	plt.show()

#--------------------------------------------------------------------------------

#TASK 3

# Principal Component Analysis or PCA is a linear feature
# extraction technique. It performs a linear mapping of the
# data to a lower-dimensional space in such a way that the
# variance of the data in the low-dimensional representation
# is maximized. It does so by calculating the eigenvectors
# from the covariance matrix. The eigenvectors that correspond
# to the largest eigenvalues (the principal components) are used
# to reconstruct a significant fraction of the variance of the
# original data.

def PCA_model (text):

	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)

	# remove stop words, punctuations
	filtered_sentence = rm_stop_words (text)
	doc = " ".join(str(x) for x in filtered_sentence)

	# Word2Vec & PCA
	word_vector = [nlp(word).vector for word in doc]
	pca.fit(word_vector)
	word_vecs_2d = pca.transform(word_vector)

	# plots
	plt.figure()
	plt.title("Principle Component Analysis (PCA) of Word2Vec", fontsize=17)
	plt.style.use('seaborn-whitegrid')
	plt.xlabel('pc1', fontsize=15)
	plt.ylabel('pc2', fontsize=15)

	# plot the scatter plot of where the words will be after PCA
	# for each word and coordinate pair: draw the text on the plot
	for word, coord in zip(filtered_sentence, word_vecs_2d):
		x, y = coord
		plt.scatter(x, y, marker='o', c='g', alpha=0.3, cmap='viridis')
		print (x,y,word)
		plt.text(x, y, word)

	#-----------------------------------------

	# Sentence2Vec & PCA
	sent_vector = [nlp(sentence.text).vector for sentence in nlp(text).sents]
	pca.fit(sent_vector)
	sent_vecs_2d = pca.transform(sent_vector)

	# plots
	plt.figure()
	plt.title("Principle Component Analysis (PCA) of Sentence2Vec", fontsize=17)
	plt.style.use('seaborn-whitegrid')
	plt.xlabel('pc1', fontsize=15)
	plt.ylabel('pc2', fontsize=15)

	# plot the scatter plot of where the sentences will be after PCA
	# and for each sentence and the coordinate pair: draw the text on the plot
	N=0
	for sentence, coord in zip(nlp(text).sents, sent_vecs_2d):
		N += 1
		x, y = coord
		plt.scatter(x, y, marker='o', c='g', alpha=0.3, cmap='viridis')
		print (WARN+sentence.text+ENDC, "PCA coordinates:", x,y)
		plt.text(x, y, sentence.text)


	# cosine similarities between sentences
	sent_similar = np.array([sent1.similarity(sent2) for sent2 in nlp(text).sents for sent1 in nlp(text).sents]).reshape(N, -1)

	# heatmap of the cosine-similarities b/w sentences
	plt.figure()
	plt.title("Cosine-similarities between the sentences", fontsize=17)
	sns.heatmap(sent_similar, annot=True, cmap="YlGnBu")

	# show the plot
	plt.show()

#--------------------------------------------------------------------------------

if __name__ == "__main__" :
	test_text = open("example1.txt").read()
	text = open("example2.txt").read()
	

	PCA_model (text)

	#token_similarity (test_text)
