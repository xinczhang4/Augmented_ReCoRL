#from __future__ import print_function
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from nltk.translate.bleu_score import sentence_bleu
# from pycocoevalcap.spice.spice import Spice

import sys
import pickle

def score_func(ref, hypo, idx=None):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        # (Spice(), "SPICE"),
        
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Meteor(),"METEOR")
    ]
    final_scores = {}
    if idx is not None:
        scorers = [scorers[idx]]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        print('score', method, score)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def self_bleu(res):
    
    self_bleu_scores = []
    
    imgIds = res.keys()

    for id in imgIds:
        hypo = res[id]

        # Sanity check.
        assert(type(hypo) is list)
        assert(len(hypo) == 1)
        

        text = hypo[0]
        sentences = text.split(". ")

        # Remove any leading or trailing whitespace from the sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
        for i, sent in enumerate(sentences):
            other_sent = [sent for j, sent in enumerate(sentences) if j != i]  # Exclude the current sentence
            bleu_score = sentence_bleu(other_sent, sent)
            self_bleu_scores.append(bleu_score)
            
    average_self_bleu = sum(self_bleu_scores) / len(self_bleu_scores)

    # print("Self-BLEU Scores:", self_bleu_scores)
    print("Average Self-BLEU Score:", average_self_bleu)

if __name__ == '__main__':
    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    print(ref_file)
    print(hyp_file)

    refs = pickle.load(open(ref_file, 'rb'))[3]
    print('len(refs)', len(refs))

    hyps = {}
    for line in open(hyp_file, 'r'):
        key, hyp = line.strip().split('\t')
        if key not in hyps:
            hyps[key] = [hyp.strip()]
    print('len(hyps)', len(hyps))
    print('keys', set(hyps.keys()) == set(refs.keys()))

    scores = score_func(refs, hyps)
    print(scores)
    
    self_bleu(hyps)
    
    



