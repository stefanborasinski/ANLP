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

    def __init__(self, mode, training_algorithm, vector_from,scc_reader,kwargdict,verbose=True):
        self.mode = mode.lower()
        if training_algorithm is not None:
            if "skip" in training_algorithm.lower():
                self.training_algorithm = 1
            else:
                self.training_algorithm = 0
        self.vector_from = vector_from.lower()
        self.scc = scc_reader
        self.verbose = verbose
        self.kwargs = kwargdict
        self.training_dir = kwargdict.get('training_dir',None)
        self.files = kwargdict.get('files',None)
        self.embedding = None
        
        if self.vector_from != "scratch":
            if self.mode == "word2vec":
                self.embedding = gensim.models.KeyedVectors.load_word2vec_format(self.kwargs['embfilepath'], binary=True)
                self.dim = self.embedding['word'].size
                self.processing_func = self._word2vec
            else:
                self.embedding = gensim.models.fasttext.FastText.load_fasttext_format(self.kwargs['embfilepath'])
                self.processing_func = self._fasttext
                if "fine" in self.vector_from:
                    self.train()
        else:
            self.train()
            
            

    def train(self):
        if len(self.files)<len(os.listdir(self.training_dir)):
            _subdivide_training()
        if self.mode == "word2vec":
               self.embedding = gensim.models.Word2Vec(gensim.models.Word2Vec.PathLineSentences(self.training_dir),size=self.kwargs.get('size',300),window=self.kwargs.get('window',10),min_count=self.kwargs.get('min_count',3),workers=multiprocessing.cpu_count(),sg=self.training_algorithm)
        else:
            if self.embedding is None:
                self.embedding = gensim.models.FastText(gensim.models.Word2Vec.PathLineSentences(self.training_dir),size=self.kwargs.get('size',300),window=self.kwargs.get('window',10),min_count=self.kwargs.get('min_count',3),workers=multiprocessing.cpu_count(),sg=self.training_algorithm)
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
        os.system("rm -rf *subdir*")
        

    def _subdivide_training(self):
        
        os.chdir(self.training_dir)
        cwd = os.getcwd()
        subdirstr = f'subdir_{len(self.files)}' 
        os.mkdir(subdirstr)
        fullsubdirpath = cwd+'/'+subdirstr
        for file in enumerate(self.files,1):
            os.system(f"cp -r {file} {fullsubdirpath}")

        self.training_dir = fullsubdirpath  
        
    def test(self, processedfiles=None):
        acc = 0
        correct, incorrect = [], []
        for question in self.scc.questions:
            q = question.get_field("question")
            translator = str.maketrans('', '', string.punctuation)
            q = q.translate(translator)  # strip q of punc, including ____ missing word space
            tokens = tokenize(q)

            #  get word2vec embedding of tokens in question (excluding answer token)
            q_vec = self.get_wordvec(tokens)

            #  calculate total word similarity by summing distances between answer token vector and question token vectors
            scores = []
            candidates = [question.get_field(ak) for ak in self.scc.keys]  # get answers as strings
            cand_vecs = self.get_wordvec(candidates)
            for ans_vec in cand_vecs:
                s = self.total_similarity(ans_vec, q_vec)
                scores.append(s)
            maxs = max(scores)
            idx = np.random.choice(
                [i for i, j in enumerate(scores) if j == maxs])  # find index/indices of answers with max score
            answer = self.scc.keys[idx][0]  # answer is first letter of key w/o accompanying bracket
            qid = question.get_field("id")
            outcome = answer == question.get_field("answer")
            if outcome:
                acc += 1
                correct.append(qid)
            else:
                incorrect.append(qid)
            if self.verbose:
                print(
                    f"{qid}: {answer} {outcome} | {question.make_sentence(question.get_field(self.scc.keys[idx]), highlight=True)}")
        log_results(self.__str__(processedfiles), acc, len(self.scc.questions), correct, incorrect, failwords=self.oovwords)

    def _word2vec(self, word, word_vec):
        if word in self.embedding:
            word_vec.append(self.embedding[word])
        else:
            word_vec.append(
                np.random.uniform(-0.25, 0.25, self.dim))  # if word not in embedding then randomly output its vector
            if word not in self.oovwords:
                self.oovwords.append(word)
        return word_vec

    def _fasttext(self, word, word_vec):
        word_vec.append(self.embedding[word])
        if word not in self.embedding.wv.vocab and word not in self.oovwords:
            self.oovwords.append(word)
        return word_vec

    def get_wordvec(self, words):
        word_vec = []
        for word in words:
            word_vec = self.processing_func(word, word_vec)
        return word_vec

    def total_similarity(self, vec, q_vec):
        score = 0
        for v in q_vec:
            score += (1 - spatial.distance.cosine(vec, v))
        # sum score of distance between vector for every word in question and vector for answer
        return score


if __name__ == '__main__':
    parser = get_default_argument_parser()
    parser.add_argument('-ta', '--training_algorithm', default=None, type=str,
                        help="learn word vectors via 'skip-gram' or 'cbow' architecture")
    parser.add_argument('-vf', '--vector_from', default=None, type=str,
                        help="how the word vector is obtained: either from 'scratch','pretrained' or 'finetuned'")
    args = parser.parse_args()
    start = time.time()
    config = load_json(args.config)
    if args.vector_from != "pretrained":
        training, _ = get_training_testing(config[args.mode]['training_dir'], split=1)
        if args.max_files is not None:
            training = training[:args.max_files]
        config['files'] = training
    scc = scc_reader()
    print(f'Loading model...')
    lm = LanguageModel(mode=args.mode, training_algorithm=args.training_algorithm, vector_from=args.vector_from, verbose=args.verbose, scc_reader=scc, kwargdict=config)
    print("Answering questions...")
    lm.test()
    endtime = time.time() - start
    print(f"Total run time: {endtime:.1f}s, {endtime / 60:.1f}m")
    os.system(
        f"cd /content/ANLP && git add -A && git commit -m 'added {lm.mode+'_'+lm.vector_from+'_'+len(lm.files)} to results.log' && git push origin master")
