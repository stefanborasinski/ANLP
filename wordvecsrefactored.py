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
                    self.train(self.kwargs['test_after'])
        else:
            self.train(self.kwargs['test_after'])
            
            

    def train(self,test_after):
        subdirs = self.subdivide_training(test_after) if test_after is not None
        if self.embedding is None:
            if self.mode == "word2vec":
                    
    
    def subdivide_training(self,corpus_split):
        fullsubdirpaths = []
        divide_after = np.ceil(len(self.files)*corpus_split) if corpus_split < 1 else divide after = corpus_split
        os.chdir(self.training_dir)
        cwd = os.getcwd()
        subdirstr = 'subdir_0' 
        os.mkdir(subdirstr)
        fullsubdirpath = cwd+'/'+subdirstr
        fullsubdirpaths.append(fullsubdirpath)
        for i, file in enumerate(self.files,1):
            if i % divide_after == 0:
                subdirstr = f'subdir_{i}' 
                os.mkdir(subdirstr)
                fullsubdirpath = cwd+'/'+subdirstr
                fullsubdirpaths.append(fullsubdirpath)
            os.system(f"cp -r {file} {fullsubdirpath}")
        
        return fullsubdirpaths
            
        
        