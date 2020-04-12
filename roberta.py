import numpy as np
import string, time
from scc import *
from utils import *
import torch

if __name__ == '__main__':
    
    parser = get_default_argument_parser()
    args = parser.parse_args()
    print(f'Loading roberta')
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    scc = scc_reader()

    guessed = []
    acc = 0
    correct, incorrect = [], []
    topk = 50000
    start = time.time()

    for question in scc.questions:
        q = question.get_field("question").replace("_____", "TEMPMASK")
        translator = str.maketrans('', '', string.punctuation)
        q = q.translate(translator)
        q = q.replace("TEMPMASK", "<mask>")
        rob_masks = roberta.fill_mask(q, topk=topk)
        rob_ranks = [mask[2] for mask in rob_masks]
        candidates = [question.get_field(ak) for ak in scc.keys]
        ans_ranks = []
        for i, candidate in enumerate(candidates):
            if candidate in rob_ranks:
                ans_ranks.append(rob_ranks.index(candidate))
            else:
                ans_ranks.append(topk)
        mins = min(ans_ranks)
        idx = np.random.choice([i for i, j in enumerate(ans_ranks) if j == mins])
        answer = scc.keys[idx][0]
        qid = question.get_field('id')
        outcome = answer == question.get_field("answer")
        if outcome:
            acc += 1
            correct.append(qid)
        else:
            incorrect.append(qid)
        if len(set(ans_ranks))==1:
            guessed.append(qid)
        if args.verbose:
            print(f"{qid}: {answer} {outcome} | {question.make_sentence(question.get_field(scc.keys[idx]), highlight=True)}")
    log_results(roberta.__str__(), acc, len(scc.questions), correct, incorrect, guessed=guessed)
    endtime = time.time() - start
    print(f"Total run time: {endtime:.1f}s, {endtime / 60:.1f}m")