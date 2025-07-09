import spacy
from nltk import SnowballStemmer

nlp = spacy.load("en_core_web_trf")
stemmer = SnowballStemmer("english")
