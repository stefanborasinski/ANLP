import os, math, argparse, time
import numpy as np
from scc import *
from utils import *
from nltk.tokenize import word_tokenize as tokenize


class LanguageModel:

    def __init__(self, trainingdir, files=[], verbose=False):
        self.training_dir = trainingdir
        self.files = files
        self.verbose = verbose
        self.train()

    def __str__(self):
        return f"ngram trained on {len(self.files)} files"

    def train(self):
        self.unigram = {}
        self.bigram = {}

        self._processfiles()
        self._make_unknowns()
        self._discount()
        self._convert_to_probs()

    def _processline(self, line):
        tokens = ["__START"] + tokenize(line) + ["__END"]
        previous = "__END"
        for token in tokens:
            self.unigram[token] = self.unigram.get(token, 0) + 1
            current = self.bigram.get(previous, {})
            current[token] = current.get(token, 0) + 1
            self.bigram[previous] = current
            previous = token

    def _processfiles(self):
        for i, afile in enumerate(self.files):
            if self.verbose:
                print(f"{i+1}/{len(self.files)} Processing {afile}")
            try:
                with open(os.path.join(self.training_dir, afile)) as instream:
                    for line in instream:
                        line = line.rstrip()
                        if len(line) > 0:
                            self._processline(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring rest of file".format(afile))

    def _convert_to_probs(self):

        self.unigram = {k: v / sum(self.unigram.values()) for (k, v) in self.unigram.items()}
        self.bigram = {key: {k: v / sum(adict.values()) for (k, v) in adict.items()} for (key, adict) in
                       self.bigram.items()}
        self.kn = {k: v / sum(self.kn.values()) for (k, v) in self.kn.items()}

    def get_prob(self, token, context="", methodparams={}):
        if methodparams.get("method", "unigram") == "unigram":
            return self.unigram.get(token, self.unigram.get("__UNK", 0))
        else:
            if methodparams.get("smoothing", "kneser-ney") == "kneser-ney":
                unidist = self.kn
            else:
                unidist = self.unigram
            bigram = self.bigram.get(context[-1], self.bigram.get("__UNK", {}))
            big_p = bigram.get(token, bigram.get("__UNK", 0))
            lmbda = bigram["__DISCOUNT"]
            uni_p = unidist.get(token, unidist.get("__UNK", 0))
            # print(big_p,lmbda,uni_p)
            p = big_p + lmbda * uni_p
            return p

    def compute_prob_line(self, line, methodparams={}):
        # this will add _start to the beginning of a line of text
        # compute the probability of the line according to the desired model
        # and returns probability together with number of tokens

        tokens = ["__START"] + tokenize(line) + ["__END"]
        acc = 0
        for i, token in enumerate(tokens[1:]):
            acc += math.log(self.get_prob(token, tokens[:i + 1], methodparams))
        return acc, len(tokens[1:])

    def _make_unknowns(self, known=2):
        unknown = 0
        for (k, v) in list(self.unigram.items()):
            if v < known:
                del self.unigram[k]
                self.unigram["__UNK"] = self.unigram.get("__UNK", 0) + v
        for (k, adict) in list(self.bigram.items()):
            for (kk, v) in list(adict.items()):
                isknown = self.unigram.get(kk, 0)
                if isknown == 0:
                    adict["__UNK"] = adict.get("__UNK", 0) + v
                    del adict[kk]
            isknown = self.unigram.get(k, 0)
            if isknown == 0:
                del self.bigram[k]
                current = self.bigram.get("__UNK", {})
                current.update(adict)
                self.bigram["__UNK"] = current

            else:
                self.bigram[k] = adict

    def _discount(self, discount=0.75):
        # discount each bigram count by a small fixed amount
        self.bigram = {k: {kk: value - discount for (kk, value) in adict.items()} for (k, adict) in self.bigram.items()}

        # for each word, store the total amount of the discount so that the total is the same
        # i.e., so we are reserving this as probability mass
        for k in self.bigram.keys():
            lamb = len(self.bigram[k])
            self.bigram[k]["__DISCOUNT"] = lamb * discount

        # work out kneser-ney unigram probabilities
        # count the number of contexts each word has been seen in
        self.kn = {}
        for (k, adict) in self.bigram.items():
            for kk in adict.keys():
                self.kn[kk] = self.kn.get(kk, 0) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='unigram or bigram ngram model, add kn if smoothing')
    parser.add_argument('-c', '--config', default='config.json', help='language model config json')
    parser.add_argument('-td', '--training_dir', type=str, default=r"data/Holmes_Training_Data",
                        help='location of Holmes_Training_Data folder')
    parser.add_argument('-mf', '--max_files', type=int, default=None, help='maximum number of files used in training. ')
    parser.add_argument('-vb', "--verbose", type=bool, default=False, help="Print out processed files alongside answers to question to debug")
    args = parser.parse_args()
    config = load_json(args.config)
    try:
        args.mode.split(" ")[1]
        mp = {k: v for k, v in config[args.mode.split(" ")[0]].items() if "smoothing" in k}
    except IndexError:
        mp = {k: v for k, v in config[args.mode].items() if "method" in k}
    if len(mp) == 0:
        print(f"Method parameters not found for {args.mode}. Terminating...")
        quit()
    training, _ = get_training_testing(split=1)
    start = time.time()
    print(f"Training {args.mode} on {args.max_files if args.max_files is not None else len(training)} files.")
    ngram = LanguageModel(files=training[:args.max_files], trainingdir=args.training_dir, verbose=args.verbose)
    scc = scc_reader()
    acc = 0
    correct, incorrect = [], []
    print("Answering questions...")
    for question in scc.questions:
        scores = []
        for key in keys:
            answord = question.get_field(key)
            q = question.make_sentence(answord)
            s, _ = ngram.compute_prob_line(q, methodparams=mp)
            scores.append(s)
        maxs = max(scores)
        idx = np.random.choice(
            [i for i, j in enumerate(scores) if j == maxs])  # find index/indices of answers with max score
        answer = keys[idx][0]  # answer is first letter of key w/o accompanying bracket
        qid = question.get_field("id")
        outcome = answer == question.get_field("answer")
        if outcome:
            acc += 1
            correct.append(qid)
        else:
            incorrect.append(qid)
        if args.verbose:
            print(f"{qid}: {answer} {outcome} | {question.make_sentence(question.get_field(keys[idx]),highlight=True)}")
    log_results(args.mode + " " + ngram.__str__(), acc, len(scc.questions), correct,
                incorrect)
    endtime = time.time() - start
    print(f"Total run time: {endtime:.1f}s, {endtime/60:.1f}m")
