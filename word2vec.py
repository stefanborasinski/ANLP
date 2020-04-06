import time, string, gensim
from scipy import spatial
from scc import *
from utils import *
import numpy as np
from nltk.tokenize import word_tokenize as tokenize

rvwords = []

def word2vec(tokens, embeddings):
    dim = embeddings['word'].size
    word_vec = []
    for word in tokens:
        if word in embeddings:
            word_vec.append(embeddings[word])
        else:
            word_vec.append(
                np.random.uniform(-0.25, 0.25, dim))  # if word not in embedding then randomly output its vector
            if word not in rvwords:
                rvwords.append(word)
    return word_vec


def total_similarity(vec, q_vec):
    score = 0
    for v in q_vec:
        score += (1 - spatial.distance.cosine(vec, v))
    # sum score of distance between vector for every word in question and vector for answer
    return score


if __name__ == '__main__':
    parser = get_default_argument_parser()
    args = parser.parse_args()
    config = load_json(args.config)

    start = time.time()
    print(f'Loading pretrained embeddings: {config[args.mode]["embedding"]}')
    if args.mode[0] == "w":
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(config[args.mode]['embedding'], binary=True)
    else:
        embeddings = gensim.models.fasttext.FastText.load_fasttext_format(config[args.mode]['embedding'])
    scc = scc_reader()
    acc = 0
    correct, incorrect = [], []

    print("Answering questions...")
    for question in scc.questions:
        q = question.get_field("question")
        translator = str.maketrans('', '', string.punctuation)
        q = q.translate(translator)  # strip q of punc, including ____ missing word space
        tokens = tokenize(q)

        #  get word2vec embedding of tokens in question (excluding answer token)
        q_vec = word2vec(tokens, embeddings)

        #  calculate total word similarity by summing distances between answer token vector and question token vectors
        scores = []
        candidates = [question.get_field(ak) for ak in keys]  # get answers as strings
        cand_vecs = word2vec(candidates, embeddings)
        for ans_vec in cand_vecs:
            s = total_similarity(ans_vec, q_vec)
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
            print(
                f"{qid}: {answer} {outcome} | {question.make_sentence(question.get_field(keys[idx]), highlight=True)}")
    log_results("word2vec", acc, len(scc.questions), correct,
                incorrect, failwords=rvwords)
    endtime = time.time() - start
    print(f"Total run time: {endtime:.1f}s, {endtime / 60:.1f}m")
