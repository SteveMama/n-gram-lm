# N-GRAM Language Model for Sentence Generation

### Background

A language model estimates the probability of a sequence of words in a sentence:

$$ P(w_1, \ldots, w_N) = \prod_{i=1}^N P(w_i | w_1, \ldots, w_{i-1}) $$

The goal is to estimate the probability of a given sentence using a **Markov Assumption** to approximate the conditional probabilities:

### N-gram Approximation
We simplify the estimation using the following assumption:

$$ P(w_i | w_1, \ldots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \ldots, w_{i-1}) $$

- **Unigram Language Model**:  
  $$ P(w_i) = \frac{\text{count}(w_i)}{\sum_{j \in V} \text{count}(w_j)} $$

- **Bigram Language Model**:  
  $$ P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})} $$

- **Trigram Language Model**:  
  $$ P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i)}{\text{count}(w_{i-2}, w_{i-1})} $$

### Smoothing
To handle unseen words or n-grams in the test data, **Laplace Smoothing** is often used:

$$ P(w_i | w_{i-n+1}, \ldots, w_{i-1}) = \frac{\text{count}(w_{i-n+1}, \ldots, w_{i-1}, w_i) + 1}{\text{count}(w_{i-n+1}, \ldots, w_{i-1}) + |V|} $$

where \( |V| \) is the size of the vocabulary.

### Perplexity
To evaluate the performance of our model, we calculate the **perplexity** on a test set:

$$ PP(w_1, \ldots, w_N) = P(w_1, \ldots, w_N)^{-\frac{1}{N}} $$

This metric measures how well the model predicts a sample, with lower perplexity indicating a better fit.

## Implementation Steps

1. **Data Preprocessing**:
   - Tokenize the input text data.
   - Create n-gram counts for each word sequence.
   
2. **Probability Estimation**:
   - Compute the probability of each n-gram using maximum likelihood estimation (MLE).
   
3. **Smoothing**:
   - Apply Laplace Smoothing to handle zero-frequency n-grams.
   
4. **Perplexity Calculation**:
   - Calculate the perplexity on the test set to evaluate the model.
   
## Example
For a **bigram** model, the training entails estimating:

$$ P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})} $$

The test perplexity is given by:

$$ PP(w_1, \ldots, w_N) = \left( \prod_{i=1}^N P(w_i | w_{i-1}) \right)^{-\frac{1}{N}} $$

## Usage

The model can be implemented in Python, using libraries like `nltk` for text preprocessing and `collections` for managing n-gram counts. Key steps include:
- **Tokenization**
- **Building n-gram Counts**
- **Probability Estimation**
- **Model Evaluation** using perplexity.


- Running the following will provide scores for Unigram, Bigram. 

```python
    python process.py
```
- To train a 'n' gram model, modify the 'n' value in the ```train_test_lm()``` function as below:

```python
    n_gram_lm = train_test_lm(n, bool)
```