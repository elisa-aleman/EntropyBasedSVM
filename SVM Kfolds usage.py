import os.path
import sys
PythonPath = os.path.join("/Volumes/GoogleDrive", "My Drive","Personal/Escuela/技大/野中研究室/Master/python/libraries") 
sys.path.append(os.path.abspath(PythonPath))
from SVMKfolds import *

k = 4
sentences = [
	['this is a brown apple and smell bad', 0],
	['i love this tasty red apple', 1],
	['this apple is small and has no taste', 0],
	['i love when the apple is juicy', 1],
	['dry apple has no value', 0],
	['sweet apple is best apple', 1],
	['sweet sweet sweet apple apple apple yes yes yes', 1],
	['i like sweet apple with red and green', 1]
]
dictionary = ['brown','red','apple','smell','green','juicy', 'tasty','dry','sweet','yes','taste','love']
SVMKFolds(k, sentences, dictionary, kernel = 'linear', C = 1.0, gamma = 0.001, times = 1)