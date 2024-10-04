from model import NGRAM_LM, Counter
import numpy as np
from typing import List, Text

def score_testfile(lm: NGRAM_LM, test_file_path: Text):
    """
    Compute the probability score for a set of sentences on a test set.
    Prints the number of sentences, average probability, and standard deviation.
    """
    with open(test_file_path, "r", encoding="utf8") as f:
        scores = [lm.score(s.strip()) for s in f.readlines()]

    print("Number of sentences:", len(scores))
    print("Average score:", np.average(scores))
    print("Std deviation:", np.std(scores))
    print()


def train_test_lm(n_gram: int, smooth: bool):
    """
    Train and test an n-gram language model.
    """
    trainingFilePath = "LM-training.txt"
    test_file_path = "LM-test.txt"
    test_sentence = "sam i am and today I am walking away"

    language_model = NGRAM_LM(n_gram, smooth)
    language_model.train(trainingFilePath)

    print("Score on test file")
    score_testfile(language_model, test_file_path)
    print("Probability of test sentence: ", language_model.score(test_sentence))
    print("Perplexity of test sentence: ", language_model.perplexity(test_sentence))

    return language_model


print("--------Bigram w/o smoothing------")
bigram_lm = train_test_lm(2, False)

print("--------Bigram w smoothing--------")
bigram_lm_smooth = train_test_lm(2, True)