import time, string, os, datetime
import gensim
from gensim.test.utils import get_tmpfile
from scipy import spatial
from scc import *
from utils import *
import numpy as np
from nltk.tokenize import word_tokenize as tokenize
import pdb


class TxtIter(object):

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with gensim.utils.open(self.filepath, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(line))


class LanguageModel:

    def __init__(self, mode, embfilepath, training_dir=None, files=[], verbose=False, checkpoint_after=None,
                 test_after=False, scc_reader=None):
        self.mode = mode
        self.oovwords = []
        self.verbose = verbose
        self.checkpoint_after = checkpoint_after
        self.test_after = test_after
        self.scc = scc_reader
        if mode == "word2vec":
            self.embedding = gensim.models.KeyedVectors.load_word2vec_format(embfilepath, binary=True)
            self.dim = self.embedding['word'].size
            self.func = self._word2vec
        if mode == "fasttext":
            self.embedding = gensim.models.fasttext.FastText.load_fasttext_format(embfilepath)
            self.func = self._fasttext
            if training_dir is not None and len(files) > 0:
                self.training_dir = training_dir
                self.files = files
                self.train()

    def __str__(self, processedfiles=None):
        if not processedfiles:
            return f"{self.mode} trained on {len(self.files) if self.files is not None else 0} files"
        else:
            return f"{self.mode} trained on {processedfiles} files"

    def train(self):
        for i, afile in enumerate(self.files):

            if self.verbose:
                print(f"{i + 1}/{len(self.files)} Processing {afile}")
            filepath = os.path.join(self.training_dir, afile)
            try:
                num_lines = sum(1 for line in open(filepath))
                txtfile = TxtIter(filepath)
                self.embedding.build_vocab(sentences=txtfile, update=True)
                self.embedding.train(sentences=txtfile, total_examples=num_lines, epochs=self.embedding.epochs)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring rest of file".format(afile))
            if self.checkpoint_after and ((i + 1) % self.checkpoint_after == 0 or i + 1 == len(self.files)):
                fname = get_tmpfile(f"fasttext_{i + 1}.model")
                print(f"Saving to disk under {fname} after training on {i + 1} files")
                self.embedding.save(fname)
                cwd = os.getcwd()
                print("tarring...")
                os.chdir('/tmp')
                os.system(f"tar -zcvf fasttext_{i + 1}.tar.gz *fasttext_{i + 1}.model* --remove-files")
                print("splitting...")
                os.system(
                    f"split -b 4000M fasttext_{i + 1}.tar.gz 'fasttext_{i + 1}.part' && rm -rf fasttext_{i + 1}.tar.gz")
                print("uploading and saving link to gdrive...")
                for f in sorted(os.listdir()):
                    if f'fasttext_{i + 1}' in f:
                        os.system(f"echo '{str(datetime.datetime.now()) + ': ' + f}' >> '/content/ANLP/linklist.txt' ")
                        os.system(f"file.io {f} >> '/content/ANLP/linklist.txt' && rm -rf {f}")
                os.chdir(cwd)
                if self.test_after:
                    self.test(i + 1)
                    os.system(
                        f"cd /content/ANLP && git add -A && git commit -m 'added fasstext_{i + 1} to results.log' && git push origin master")

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
            word_vec = self.func(word, word_vec)
        return word_vec

    def total_similarity(self, vec, q_vec):
        score = 0
        for v in q_vec:
            score += (1 - spatial.distance.cosine(vec, v))
        # sum score of distance between vector for every word in question and vector for answer
        return score


if __name__ == '__main__':
    parser = get_default_argument_parser()
    parser.add_argument('-ca', '--checkpoint_after', default=None, type=int,
                        help='number of files processed after which to create a checkpoint')
    parser.add_argument('-ta', '--test_after', default=None, type=bool,
                        help='whether or not to test after checkpointing')
    start = time.time()
    args = parser.parse_args()
    config = load_json(args.config)
    training, _ = get_training_testing(config[args.mode]['training_dir'], split=1)
    if args.max_files is not None:
        training = training[:args.max_files]
    args.files = training
    scc = scc_reader()
    print(f'Loading pretrained embeddings: {config[args.mode]["embfilepath"]}')
    lm = LanguageModel(args.mode, files=args.files, verbose=args.verbose, checkpoint_after=args.checkpoint_after,
                       test_after=args.test_after, scc_reader=scc, **config[args.mode])
    if not args.test_after:
        print("Answering questions...")
        lm.test()
        endtime = time.time() - start
        print(f"Total run time: {endtime:.1f}s, {endtime / 60:.1f}m")
