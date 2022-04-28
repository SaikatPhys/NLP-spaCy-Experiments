HEAD = '\033[95m'#magenta
OKB = '\033[94m'#blue
OKG = '\033[92m'#green
WARN = '\033[93m'#yellow ochre
FAIL = '\033[91m'#red
ENDC = '\033[0m'#ends an effect. So, use like HEAD+ULINE+"Hello"+ENDC+ENDC
BOLD = '\033[1m'
ULINE = '\033[4m'

import numpy as np
import string

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
# And lemmatize the words.

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
	#filtered_sent = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
	filtered_sent = [word.lemma_ for word in doc]
	filtered_sent = [word for word in filtered_sent if word not in spacy_stopwords and word not in punctuations]
	filtered_sentence = " ".join(str(x) for x in filtered_sent)

	#print(FAIL+"Filtered sentence after removing the stop words, punctuations and made lower-case :"+ENDC)
	#print (len(filtered_sent))
	#print (filtered_sent)
	#print("---------------------")
	return filtered_sentence


#--------------------------------------------------------------------------------

# TASK 2

# Rule-Based named entity recognition

def rule_based_entities (text):
	from spacy.pipeline import EntityRuler
	ruler = EntityRuler(nlp)

	text = rm_stop_words (text)

	patterns = [{"label": "PHYS", "pattern": [{"lower":"acceleration"}]},
				{"label": "PHYS", "pattern": [{"lower":"accretion"}]},
				{"label": "PHYS", "pattern": [{"lower": "accretion"}, {"lower": "disks"}]},
				{"label": "PHYS", "pattern": [{"lower":"asteroseismology"}]},
				{"label": "PHYS", "pattern": [{"lower":"astrobiology"}]},
				{"label": "PHYS", "pattern": [{"lower":"astrochemistry"}]},
				{"label": "PHYS", "pattern": [{"lower": "astroparticle"}, {"lower": "physics"}]},
				{"label": "PHYS", "pattern": [{"lower": "atomic"}, {"lower": "data"}]},
				{"label": "PHYS", "pattern": [{"lower": "atomic"}, {"lower": "processes"}]},
				{"label": "PHYS", "pattern": [{"lower":"blackhole"}]},
				{"label": "PHYS", "pattern": [{"lower": "black"}, {"lower": "hole"}]},
				{"label": "PHYS", "pattern": [{"lower": "black"}, {"lower": "hole"}, {"lower": "physics"}]},
				{"label": "PHYS", "pattern": [{"lower":"chaos"}]},
				{"label": "PHYS", "pattern": [{"lower":"conduction"}]},
				{"label": "PHYS", "pattern": [{"lower":"convection"}]},
				{"label": "PHYS", "pattern": [{"lower": "dense"}, {"lower": "matter"}]},
				{"label": "PHYS", "pattern": [{"lower":"diffusion"}]},
				{"label": "PHYS", "pattern": [{"lower":"dynamo"}]},
				{"label": "PHYS", "pattern": [{"lower": "elementary"}, {"lower": "particles"}]},
				{"label": "PHYS", "pattern": [{"lower": "equation"}, {"lower": "of"}, {"lower": "state"}]},
				{"label": "PHYS", "pattern": [{"lower":"gravitation"}]},
				{"label": "PHYS", "pattern": [{"lower": "gravitational"}, {"lower": "lensing"}]},
				{"label": "PHYS", "pattern": [{"lower": {"IN": ["strong", "weak", "micro"]}}, {"lower": "gravitational"}, {"lower": "lensing"}]},
				{"label": "PHYS", "pattern": [{"lower": "gravitational"}, {"lower": "waves"}]},
				{"label": "PHYS", "pattern": [{"lower":"hydrodynamics"}]},
				{"label": "PHYS", "pattern": [{"lower":"instabilities"}]},
				{"label": "PHYS", "pattern": [{"lower": "line"}, {"lower": {"IN": ["formation", "identification", "profiles"]}}]},
				{"label": "PHYS", "pattern": [{"lower": "magnetic"}, {"lower": {"IN": ["fields", "reconnection"]}}]},
				{"label": "PHYS", "pattern": [{"ORTH":"MHD"}]},
				{"label": "PHYS", "pattern": [{"lower":"magnetohydrodynamics"}]},
				{"label": "PHYS", "pattern": [{"lower":"masers"}]},
				{"label": "PHYS", "pattern": [{"lower": "molecular"}, {"lower": {"IN": ["data", "processes"]}}]},
				{"label": "PHYS", "pattern": [{"lower":"neutrinos"}]},
				{"label": "PHYS", "pattern": [{"lower":"nucleosynthesis"}]},
				{"label": "PHYS", "pattern": [{"lower":"abundances"}]},
				{"label": "PHYS", "pattern": [{"lower": "nuclear"}, {"lower": {"IN": ["reactions", "abundances"]}}]},
				{"label": "PHYS", "pattern": [{"lower":"opacity"}]},
				{"label": "PHYS", "pattern": [{"lower":"plasmas"}]},
				{"label": "PHYS", "pattern": [{"lower":"polarization"}]},
				{"label": "PHYS", "pattern": [{"lower": "radiation"}, {"lower": {"IN": ["dynamics", "mechanisms", "transfer"]}}]},
				{"label": "PHYS", "pattern": [{"lower": "radiative"}, {"lower": "transfer"}]},
				{"label": "PHYS", "pattern": [{"lower": "relativistic"}, {"lower": "processes"}]},
				{"label": "PHYS", "pattern": [{"lower":"scattering"}]},
				{"label": "PHYS", "pattern": [{"lower": "shock"}, {"lower": "waves"}]},
				{"label": "PHYS", "pattern": [{"lower": "solid"}, {"lower": "state"}]},
				{"label": "PHYS", "pattern": [{"lower":"turbulence"}]},
				{"label": "PHYS", "pattern": [{"lower":"waves"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower":"Cosmology"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "Hot"}, {"lower": "Big"}, {"lower": "Bang"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "Big"}, {"lower": "Bang"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "Big"}, {"lower": "Bang"}, {"lower": "Cosmology"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "Cosmic"}, {"lower": "Background"}, {"lower": "Radiation"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "Cosmic"}, {"lower": "Microwave"}, {"lower": "Background"}, {"lower": "Radiation"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "Microwave"}, {"lower": "Background"}, {"lower": "Radiation"}]},
				{"label": "COSMOLOGY", "pattern": [{"ORTH":"CMB"}]},
				{"label": "COSMOLOGY", "pattern": [{"ORTH":"CMBR"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "cosmological"}, {"lower": "parameters"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "dark"}, {"lower": "ages"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "dark"}, {"lower": "matter"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "dark"}, {"lower": "energy"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "diffuse"}, {"lower": "radiation"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "distance"}, {"lower": "scale"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "early"}, {"lower": "Universe"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "inflation"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "reionization"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "large"}, {"lower": "scale"}, {"lower": "structures"}]},
				{"label": "COSMOLOGY", "pattern": [{"lower": "primordial"}, {"lower" : "nucleosynthesis"}]}]

	#EntityRuler = ruler.from_disk("./ASTRO_PATTERNS.jsonl")

	ruler.add_patterns(patterns)
	nlp.add_pipe(ruler)

	print(OKG+BOLD+"Named Entity Recognition (NER) or Entity extraction using Natural Language Processing (NLP)."+ENDC+ENDC)
	print(OKG+BOLD+"Process of NER tags sequences of words in a given text as a person, org., place, topic, etc."+ENDC+ENDC)
	print("---------------------")
	for entity in nlp(text).ents:
		print("Entity: ", FAIL+entity.text+ENDC)
		print("Entity Type: %s | %s" % (entity.label_, spacy.explain(entity.label_)))
		print("Label of the named Entity: ", entity.label)
		print("Start Offset of the Entity: ", entity.start_char)
		print("End Offset of the Entity: ", entity.end_char)
		print("---------------------")


	displacy.serve (nlp(text), style="ent")


#--------------------------------------------------------------------------------

if __name__ == "__main__" :
	text = open("example.txt").read()
	rule_based_entities (text)
