from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Text

def create_ngrams(tokens: List[Text], n: int):
    """
    Take a sequence of tokens and return a list of n-grams.
    """
    if n == 0:
        n = 1
    return [tuple(tokens[i:n+i]) for i in range(len(tokens)-n+1)]


class NGRAM_LM:
    unk = "<UNK>"
    start_token = "<s>"
    end_token = "</s>"

    def __init__(self, n_gram: int, smoothing: bool):
        # n-gram order
        self.n = n_gram
        # enable/disable Laplace smoothing
        self.smoothing = smoothing
        # counter for n-grams (i.e., the numerator to calculate n-gram probs)
        self.n_grams = Counter()
        # counter for n-1-grams (i.e., the denominator to calculate n-gram probs)
        self.n_minus_one_grams = Counter()

    def train(self, training_file_path: Text):
        """
        Train the language model by accumulating counts of n-grams and n-1-grams.
        """
        with open(training_file_path, "r") as f:
            unigrams = Counter()
            all_tokens = []

            for sentence in f:
                words = sentence.strip().split()
                unigrams.update(words)
                all_tokens += words

            unigrams = Counter([word if unigrams[word] > 1 else self.unk for word in all_tokens])
            self.vocab = list(unigrams.keys())
            self.token_counts = sum(unigrams.values())

        tokens = [word if unigrams[word] > 1 else self.unk for word in all_tokens]

        self.n_grams.update(create_ngrams(tokens, self.n))
        if self.n > 1:
            self.n_minus_one_grams.update(create_ngrams(tokens, self.n - 1))

        print("Training n-gram:", self.n)
        print("Vocab size:", len(self.vocab))
        print("Token Counts:", self.token_counts)
        print("N-gram Counts:", sum(self.n_grams.values()))
        print("Unique n-grams:", len(self.n_grams))
        print("N-1-gram Counts:", sum(self.n_minus_one_grams.values()))
        print("Unique n-1-grams:", len(self.n_minus_one_grams))
        print("<UNK> Counts:", unigrams[self.unk])
        print()

    def score(self, sentence: Text):
        """
        Compute the log probability of a sequence as a sum of individual n-gram log probabilities.
        """
        words = sentence.strip().split()
        words = [self.unk if word not in self.vocab else word for word in words]

        as_ngrams = create_ngrams(words, self.n)
        total_log_prob = 0

        for ngram in as_ngrams:
            if self.n == 1:
                count = self.n_grams[ngram]
                prob = (count + 1) / (
                            self.token_counts + len(self.vocab)) if self.smoothing else count / self.token_counts
            else:
                ngram_count = self.n_grams[ngram]
                n_prev_gram = ngram[:-1]
                n_prev_count = self.n_minus_one_grams[n_prev_gram]

                prob = (ngram_count + 1) / (n_prev_count + len(self.vocab)) if self.smoothing else ngram_count / max(
                    n_prev_count, 1)

            # Use log-probabilities for sum
            prob = max(prob, 1e-10)
            total_log_prob += np.log(prob)

        return total_log_prob

    def perplexity(self, sentence: Text):
        """
        Compute the perplexity of a sentence under the model using the correct log-probability sum.
        """
        total_log_prob = self.score(sentence)
        N = len(sentence.strip().split())
        return math.exp(-total_log_prob / N)

    def generate(self):
        """
          Generate a sentence using Shannon's method
        """
        num_begin = self.n - 1 if self.n > 1 else 1
        sent = [self.start_token for i in range(num_begin)]
        curr = sent[len(sent) - 1]
        if self.n == 1:
            # remove the <s> from our vocab for unigrams
            lookup = [word for word in self.vocab if word != self.start_token]
            # remove counts of the start of sentence
            token_counts = self.token_counts - self.n_grams[tuple([self.start_token])]
            weights = [(self.n_grams[tuple([word])]) / (token_counts) for word in lookup]

        while curr != self.end_token:
            if (self.n > 1):
                # get the n - 1 previous words that we are sampling
                previous = tuple(sent[len(sent) - (self.n - 1): len(sent)])
                previous_count = self.n_minus_one_grams[previous]

                lookup = [choice for choice in self.n_grams if choice[:-1] == previous]
                weights = [self.n_grams[choice] / previous_count for choice in lookup]

            to_sample = np.arange(len(lookup))
            next = lookup[np.random.choice(to_sample, p=weights)]
            # avoid generating just start and end of sentence
            if next == self.end_token and curr == self.start_token: continue

            if self.n == 1:
                sent.append(next)
            else:
                sent.append(next[-1])
            curr = sent[-1]

        return " ".join(sent)