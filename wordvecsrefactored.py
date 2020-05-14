import time, string, os, datetime
import gensim
from gensim.test.utils import get_tmpfile
from scipy import spatial
from scc import *
from utils import *
import numpy as np
from nltk.tokenize import word_tokenize as tokenize
import pdb
import multiprocessing

class LanguageModel:

    def __init__(self, mode, training_algorithm, vector_from,scc_reader,verbose=False,**kwargs):
        self.mode = mode.lower()
        self.training_algorithm = 1 if "skip" in training_algorithm.lower() else self.training_algorithm = 0 
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
                    self.train_and_test()
        else:
            self.train_and_test()
            
            

    def train_and_test(self):
        if len(self.files)<len(os.listdir(self.training_dir)):
            _subdivide_training()
        if self.mode == "word2vec":
               self.embedding = gensim.models.Word2Vec(gensim.models.Word2Vec.PathLineSentences(self.training_dir),size=self.kwargs.get('size',300),window=self.kwargs.get('window',10),min_count=self.kwargs.get('min_count',3),workers=multiprocessing.cpu_count(),sg=self.training_algorithm)
        else:
            if self.embedding is None:
                self.embedding = gensim.models.FastText(gensim.models.Word2Vec.PathLineSentences(self.training_dir),self.kwargs.get('size',300),self.kwargs.get('window',10),self.kwargs.get('min_count',3))
            else:
                for i, afile in enumerate(self.files):
                    if self.verbose:
                        print(f"{i + 1}/{len(self.files)} Processing {afile}")
                    filepath = os.path.join(self.training_dir, afile)
                    try:
                        num_lines = sum(1 for line in open(filepath))
                        self.embedding.build_vocab(corpus_file=filepath, update=True)
                        self.embedding.train(corpus_file=filepath, total_examples=num_lines, epochs=5)
                    except UnicodeDecodeError:
                        print("UnicodeDecodeError processing {}: ignoring rest of file".format(afile))
        
        filestr = self.mode+'_'+self.vector_from+'_'+len(self.files)
        fname = get_tmpfile(f"{filestr}.model")
        print(f"Saving to disk under {fname}")
        self.embedding.save(fname)
        cwd = os.getcwd()
        print("tarring...")
        os.chdir('/tmp')
        os.system(f"tar -zcvf {filestr}.tar.gz *{filestr}.model* --remove-files")
        print("splitting...")
        os.system(
            f"split -b 4000M {filestr}.tar.gz '{filestr}.part' && rm -rf {filestr}.tar.gz")
        print("uploading and saving link to repository...")
        for f in sorted(os.listdir()):
            if f'{filestr}' in f:
                os.system(f"echo '{str(datetime.datetime.now()) + ': ' + f}' >> '/content/ANLP/linklist.txt' ")
                os.system(f"file.io {f} >> '/content/ANLP/linklist.txt' && rm -rf {f}")
        os.chdir(cwd)
        self.test()
        os.system(
            f"cd /content/ANLP && git add -A && git commit -m 'added fasstext_{i + 1} to results.log' && git push origin master")
        

    def _subdivide_training(self):
        os.chdir(self.training_dir)
        cwd = os.getcwd()
        subdirstr = f'subdir_{len(self.files)}' 
        os.mkdir(subdirstr)
        fullsubdirpath = cwd+'/'+subdirstr
        for file in enumerate(self.files,1):
            os.system(f"cp -r {file} {fullsubdirpath}")

        self.training_dir = fullsubdirpath  
        
        