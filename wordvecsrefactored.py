import time, string, os, datetime
import gensim
from gensim.test.utils import get_tmpfile
from scipy import spatial
from scc import *
from utils import *
import numpy as np
from nltk.tokenize import word_tokenize as tokenize
import pdb

class LanguageModel:

    def __init__(self, mode, training_algorithm, vector_from,scc_reader,verbose=False,**kwargs):
        self.mode = mode.lower()
        self.training_algorith = training_algorithm.lower()
        self.vector_from = vector_from.lower()
        self.scc = scc_reader
        self.verbose = verbose
        self.kwargs = kwargs
        self.training_dir = kwargs.get('training_dir',None)
        self.files = kwargs.get('files',None)
        self.embedding = None
        
        if self.vector_from != "scratch":
            if self.mode == "word2vec":
                self.embedding = gensim.models.KeyedVectors.load_word2vec_format(self.kwargs['embfilepath'], binary=True)
            self.dim = self.embedding['word'].size
            self.processing_func = self._word2vec
            else:
                self.embedding = gensim.models.fasttext.FastText.load_fasttext_format(embfilepath)
                self.processing_func = self._fasttext
                if self.vector_from == "finetuning":
                    self.train()
        else:
            self.train()
            
            

    def train(self):
        if len(self.files)<len(os.listdir(self.training_dir)):
            _subdivide_training()
        if self.mode == "word2vec":
               self.embedding = gensim.models.Word2Vec(gensim.models.Word2Vec.PathLineSentences(self.training_dir),self.kwargs.get('size',300),self.kwargs.get('window',10),self.kwargs.get('min_count',3))
    

    def _subdivide_training(self):
        os.chdir(self.training_dir)
        cwd = os.getcwd()
        subdirstr = f'subdir_{len(self.files)}' 
        os.mkdir(subdirstr)
        fullsubdirpath = cwd+'/'+subdirstr
        for file in enumerate(self.files,1):
            os.system(f"cp -r {file} {fullsubdirpath}")

        self.training_dir = fullsubdirpath  
        
        