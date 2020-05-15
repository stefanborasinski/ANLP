import numpy as np
import string, time
from scc import *
from utils import *
import torch
import pdb

if __name__ == '__main__':
    
    parser = get_default_argument_parser()
    args = parser.parse_args()
    print(f'Loading {args.mode}')
    roberta = torch.hub.load('pytorch/fairseq', args.mode) #load either roberta.base or roberta.large
    scc = scc_reader()

    acc = 0
    correct, incorrect, guessed = [], [], []
    if args.mode == "robert.large":
        topk = 49500
    else:
        topk = 3000
    start = time.time()
    

    for question in scc.questions:
        q = question.get_field("question").replace("_____", "TEMPMASK")
        translator = str.maketrans('', '', string.punctuation)
        q = q.translate(translator) #strip question of punctuation
        q = q.replace("TEMPMASK", "<mask>") #replace space for candidate token with mask token
        rob_masks = roberta.fill_mask(q, topk=topk) #get list of top k fill mask candidates
        rob_ranks = [mask[2] for mask in rob_masks] #filter for tokens
        candidates = [question.get_field(ak) for ak in scc.keys]
        ans_ranks = []
        for i, candidate in enumerate(candidates):
            if candidate in rob_ranks:
                ans_ranks.append(rob_ranks.index(candidate)) #if candidate is found, get top k ranking
            else:
                ans_ranks.append(topk) #if not found assign lowest value, the topk parameter itself
        mins = min(ans_ranks) #filter for top ranking candidate
        idx = np.random.choice([i for i, j in enumerate(ans_ranks) if j == mins])
        answer = scc.keys[idx][0]
        qid = question.get_field('id')
        outcome = answer == question.get_field("answer")
        if outcome:
            acc += 1
            correct.append(qid)
        else:
            incorrect.append(qid)
        if len(set(ans_ranks))==1: #if no candidates were found in the top k list record a guess was made
            guessed.append(qid)
        if args.verbose:
            print(f"{qid}: {answer} {outcome} | {question.make_sentence(question.get_field(scc.keys[idx]), highlight=True)}")
    log_results("roberta", acc, len(scc.questions), correct, incorrect, guessed=guessed)
    endtime = time.time() - start
    print(f"Total run time: {endtime:.1f}s, {endtime / 60:.1f}m")