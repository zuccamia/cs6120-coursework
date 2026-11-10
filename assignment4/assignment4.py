#!/usr/bin/env python
# coding: utf-8

# # Assignment 4 - Autocomplete

# ## Setup & Processing

# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import math

import nltk
import utils

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.path.append('.')


# In[4]:


class SpecialTokens:
    """ 
    Class of special tokens
    """
    def __init__(self, start_token = "<s>", end_token = "<e>", unknown_token = "<unk>"):
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

class NGramModel():
    """
    This class holds your n-gram model and all its parameters, which include:

      - previous_n_gram: the last n-gram that appeared prior to the current one
      - n-gram_counts: a dictionary specifying the context words
    """
    def __init__(self, n_gram_counts,
               n_plus1_gram_counts, vocabulary, special_tokens, k=1):
        self.n_gram_counts = n_gram_counts
        self.n_plus1_gram_counts = n_plus1_gram_counts
        self.vocabulary = vocabulary
        self.special_tokens = special_tokens
        self.k = k


# ## Q1 Process Data

# In[5]:


def preprocess_data(filename, count_threshold, special_tokens,
                    sample_delimiter='\n', split_ratio=0.8):
    """
    Ungraded: You do not need to change this function.

    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" .
    Args:
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        training_data = list of lists denoting tokenized sentence. This looks like
                        the following:

                        [ ["this", "<unk>", "example"], 
                          ["another", "sentence", "<unk>", "right"],
                         ...
                        ] 
        test_data = Same format as above.
        vocabulary = list of vocabulary words. This looks like the following:

                        ["vocab-word-1", "vocab-word-2", etc.]
    """

    # Create sentences and tokenize the data to create a list of strings. 
    tokenized_data = read_and_tokenize_sentences(filename, sample_delimiter)

    # Create the training / test splits
    train_size = int(len(tokenized_data) * split_ratio)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    # Get the closed vocabulary using the train data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # For the train data, replace less common words with unknown token
    train_data_replaced = replace_oov_words_by_unk(
        train_data, vocabulary, unknown_token = special_tokens.unknown_token)

    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(
        test_data, vocabulary, unknown_token = special_tokens.unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary

def preprocess_data_test():
    """
    Ungraded: You can use this function to test out preprocess_data. 
    """
    tmp_train = "the sky is blue.\nleaves are green.\nsmell all the roses."
    tmp_test = "roses are red."

    with open('tmp_data.txt', 'w') as f:
      f.write(str(tmp_train) + '\n')
      f.write(str(tmp_test) + '\n')

    special_tokens = SpecialTokens()
    count_threshold = 1

    tmp_train_repl, tmp_test_repl, tmp_vocab = preprocess_data(
        "tmp_data.txt", count_threshold, special_tokens, split_ratio = 0.75)

    assert tmp_test_repl == [['roses', 'are', '<unk>', '.']] or \
      tmp_test_repl == [[special_tokens.start_token, 
                         'roses', 'are', '<unk>', 
                         special_tokens.end_token]] or \
      tmp_test_repl == [[special_tokens.start_token, 
                         'roses', 'are', '<unk>', '.',
                         special_tokens.end_token]], \
      print("tmp_test_repl is not correct")

    assert tmp_train_repl == [['the', 'sky', 'is', 'blue', '.'],
                              ['leaves', 'are', 'green', '.'],
                              ['smell', 'all', 'the', 'roses', '.']] or \
           tmp_train_repl == [[special_tokens.start_token, 
                               'the', 'sky', 'is', 'blue', 
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'leaves', 'are', 'green', 
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'smell', 'all', 'the', 'roses', 
                               special_tokens.end_token]] or \
           tmp_train_repl == [[special_tokens.start_token, 
                               'the', 'sky', 'is', 'blue', '.',
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'leaves', 'are', 'green', '.',
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'smell', 'all', 'the', 'roses', '.',
                               special_tokens.end_token]], \
      print("tmp_train_repl is not correct")

    print("\033[92m Successful test")

    return


# ### Q1.1 Read / Tokenize Data from Sentences

# In[6]:


def read_and_tokenize_sentences(filename, sample_delimiter="\n"):
    """
    Args:
        - filename = (e.g., "en_US.twitter.txt")
        - sample_delimiter = delimits each sample (i.e., each tweet)

    Example usage: 
       $ read_and_tokenize_sentences(sentences) 

       [['sky', 'is', 'blue', '.'],
        ['leaves', 'are', 'green'],
        ['roses', 'are', 'red', '.']]A

    You can use nltk's tokenize function here.

       nltk.word_tokenize(sentence)
    """
    with open(filename, "r", encoding="utf-8") as f:
        sentences = f.read().split(sample_delimiter)
    tokenized = [nltk.word_tokenize(sentence) for sentence in sentences if sentence.strip()]

    return tokenized

def get_words_with_nplus_frequency(train_data, count_threshold):
    """
    Args:
        - train_data: list of lists denoting tokenized sentence. This looks like the following:
                        [['sky', 'is', 'blue', '.'],
                        ['leaves', 'are', 'green'],
                        ['roses', 'are', 'red', '.']]
        - count_threshold: minimum frequency for a word worth modelling and adding to the vocabulary
    Returns:
        - vocabulary: list of vocabulary words. This looks like the following:
                        ["vocab-word-1", "vocab-word-2", etc.]
    """
    from collections import Counter
    freqs = Counter(word for sentence in train_data for word in sentence)
    vocabulary = [word for word, count in freqs.items() if count >= count_threshold]

    return vocabulary


# ### Q1.2 Replace OOV Words with Special Token

# In[7]:


def replace_oov_words_by_unk(data, vocabulary, unknown_token="<unk>"):
    """
    Args:
        - data: list of lists denoting tokenized sentence. This looks like the following:
                        [['sky', 'is', 'blue', '.'],
                        ['leaves', 'are', 'green'],
                        ['roses', 'are', 'red', '.']]
        - vocabulary: list of vocabulary words. This looks like the following:
                        ["vocab-word-1", "vocab-word-2", etc.]
        - unknown_token: symbol that replaces any token that are not part of the given vocabulary
    Returns:
        - unked_data: list of lists denoting tokenized sentence. This looks like
                        the following:
                        [ ["this", "<unk>", "example"], 
                          ["another", "sentence", "<unk>", "right"],
                         ...
                        ] 
    """
    unked_data = [[word if word in vocabulary else unknown_token for word in sentence] for sentence in data]

    return unked_data


# In[8]:


preprocess_data_test()


# ## Q2 Count N-Grams

# In[9]:


def count_n_grams(data, n, special_tokens):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words
        n: Number of words in a sequence
        special_tokens: A structure that contains:
          - start_token = "<s>"
          - end_token = "<e>"
          - unknown_token = "unk"

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """
    # Initialize dictionary of n-grams and their counts
    n_grams = {}
    # Pad each sentence (list of words in a sentence) with start and end tokens
    padded_data = [[special_tokens.start_token] + sentence + [special_tokens.end_token] for sentence in data]
    # Slide the window for each sentence to get tuples of n-grams
    # Map the tuple of n-grams with the n-gram tuple's frequency
    for sentence in padded_data:
        for i in range(len(sentence)-n+1):
            n_gram = tuple(sentence[i:(i+n)])
            n_grams[n_gram] = n_grams.get(n_gram, 0) + 1

    return n_grams

def count_n_grams_test():
    """
    Ungraded: You can use this function to test out count_n_grams. 
    """
    tmp_data = "i like a cat\nthis dog is like a cat"
    with open('tmp_data.txt', 'w') as f:
      f.write(tmp_data + '\n')

    sentences, _, _ = preprocess_data(
        "tmp_data.txt", 0, SpecialTokens(), split_ratio = 1.0)

    received = count_n_grams(sentences, 2, SpecialTokens())
    expected = { ('<s>', 'i'): 1,
      ('i', 'like'): 1, ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2,
      ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1, ('is', 'like'): 1}

    assert received == expected, print("Received: \n", received, 
                                       "\n\nExpected: \n", expected)

    print("\033[92m Successful test")
    return

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    """
    To use this function, you'll need to get a utils function:

    >> wget -nc https://course.ccs.neu.edu/cs6120f26/data/twitter/utils.py

    Then, you can visualize your probability matrices. For example, consider the following:

        sentences = [['i', 'like', 'a', 'cat'],
                         ['this', 'dog', 'is', 'like', 'a', 'cat']]
        unique_words = list(set(sentences[0] + sentences[1]))
        bigram_counts = count_n_grams(sentences, 2, SpecialTokens())
        display(make_probability_matrix(bigram_counts, unique_words, k=1))
    """
    count_matrix = utils.make_count_matrix(n_plus1_gram_counts, vocabulary)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1) + k*len(vocabulary), axis=0)
    return prob_matrix


# In[10]:


count_n_grams_test()

sentences = [['i', 'like', 'a', 'cat'], ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
bigram_counts = count_n_grams(sentences, 2, SpecialTokens())
print("bigram counts")
display(utils.make_count_matrix(bigram_counts, unique_words))
print("bigram probabilities")
display(make_probability_matrix(bigram_counts, unique_words, k=1))


# ## Q3 Estimate the Probabilities

# In[11]:


def next_words(ngram_model):
    special_tokens = ngram_model.special_tokens
    return ngram_model.vocabulary + [special_tokens.end_token, special_tokens.unknown_token]

def estimate_probability(word, context_tokens, ngram_model):
    # P(w | context) = (C(context, w) + k) / (C(context) + k·|V|)
    context = tuple(context_tokens)
    context_count = ngram_model.n_gram_counts.get(context, 0)
    next_word_count = ngram_model.n_plus1_gram_counts.get(context + (word,), 0)
    vocab_size = len(next_words(ngram_model))
    k = ngram_model.k

    return (next_word_count+k)/(context_count+k*vocab_size)

def estimate_probabilities(context_tokens, ngram_model):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing

    Args:
        context_tokens: A list of previous n words
        ngram_model: a structure that contains:
            - n_gram_counts: Dictionary of counts of n-grams
            - n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            - vocabulary: list of unique words
            - k: positive constant, smoothing parameter

    Returns:
        A dictionary mapping from next words to probability
    """
    probabilities = {}
    for word in next_words(ngram_model):
        probabilities[word] = estimate_probability(word, context_tokens, ngram_model)

    return probabilities

def estimate_probabilities_test():
    """
    Ungraded: You can use this function to test out estimate_probabilities. 
    """

    tmp_data = "i like a cat\nthis dog is like a cat"
    with open('tmp_data.txt', 'w') as f:
      f.write(tmp_data + '\n')

    sentences, _, vocabulary = preprocess_data(
        "tmp_data.txt", 0, SpecialTokens(), split_ratio = 1.0)

    # unique_words = list(set(sentences[0] + sentences[1]))
    unigram_counts = count_n_grams(sentences, 1, SpecialTokens())
    bigram_counts = count_n_grams(sentences, 2, SpecialTokens())

    ngram_model = NGramModel(unigram_counts, bigram_counts, vocabulary,
                             SpecialTokens(), k=1)

    expected = {'i': 0.09090909090909091, 'like': 0.09090909090909091, 
                'a': 0.09090909090909091, 'cat': 0.2727272727272727, 
                'this': 0.09090909090909091, 'dog': 0.09090909090909091, 
                'is': 0.09090909090909091, '<e>': 0.09090909090909091, 
                '<unk>': 0.09090909090909091}

    assert estimate_probabilities(["a"], ngram_model) == expected, \
      print("estimate_probabilities is not correct")

    print("\033[92m Successful test")


# In[12]:


estimate_probabilities_test()


# ## Q4 Inference

# In[13]:


def predict_next_word(partial_sentence, ngram_model):
    """
    Predicts the next word in a partial sentence using the provided n-gram model.

    Args:
        partial_sentence: The partial sentence as a string.
        ngram_model: An instance of the NGramModel class.

    Returns:
        The predicted next word (string) or None if no prediction can be made.
    """
    n = len(next(iter(ngram_model.n_gram_counts))) # first key in n_grams: a tuple of sliced n grams
    special_tokens = ngram_model.special_tokens
    # Pad so short inputs still have n tokens, then keep the last n as context
    tokens = [special_tokens.start_token]*n + nltk.word_tokenize(partial_sentence)
    context_tokens = tokens[-n:]
    # Replace unknown words with unknown token
    context_tokens = replace_oov_words_by_unk([context_tokens], ngram_model.vocabulary, special_tokens.unknown_token)[0]

    probs = estimate_probabilities(context_tokens, ngram_model)
    next_word = max(probs, key=probs.get)
    return next_word


# ## Q5 Extra Credit

# In[14]:


class StyleGram:
    """
    Classifies a passage by author style and predicts the next words,
    using one trigram model per style.

    Warning:
        Relies on the following functions from Q1 to Q3 as helpers:
        read_and_tokenize_sentences, get_words_with_nplus_frequency,
        replace_oov_words_by_unk, count_n_grams, estimate_probability,
        and next_words. 
        These must be defined before StyleGram is used.
    """
    COUNT_THRESHOLD = 3
    STYLE_N = 3
    SMOOTHING_K = 1
    SPECIAL_TOKENS = SpecialTokens()

    def __init__(self, style_files, split_ratio=0.8):
        """
        We will only be passing style_files in. All your processing and
        training should be done by the time this function returns.
        """
        self.style_files = style_files
        unk = self.SPECIAL_TOKENS.unknown_token

        # Tokenize and split each file
        all_train, all_test = [], []
        for filename in style_files:
            data = read_and_tokenize_sentences(filename, "\n")
            split = int(len(data) * split_ratio)
            all_train.append(data[:split])
            all_test.append(data[split:])

        # One shared vocabulary for all styles
        self.vocabulary = get_words_with_nplus_frequency(
            [sentence for train in all_train for sentence in train], self.COUNT_THRESHOLD)
        self.test_data = [replace_oov_words_by_unk(test, self.vocabulary, unk) for test in all_test]

        # One trigram model per style, with context counts derived from the trigrams
        self.models = []
        for train in all_train:
            train = replace_oov_words_by_unk(train, self.vocabulary, unk)
            ngram_counts = count_n_grams(train, self.STYLE_N, self.SPECIAL_TOKENS)
            context_counts = {}
            for ngram, count in ngram_counts.items():
                context_counts[ngram[:-1]] = context_counts.get(ngram[:-1], 0) + count
            self.models.append(NGramModel(context_counts, ngram_counts, self.vocabulary, self.SPECIAL_TOKENS, k=self.SMOOTHING_K))

    def _prepare(self, sentences):
        # Replace unknown words, then pad each sentence with (n - 1) start tokens
        sentences = replace_oov_words_by_unk(sentences, self.vocabulary,
                                             self.SPECIAL_TOKENS.unknown_token)
        padding = [self.SPECIAL_TOKENS.start_token] * (self.STYLE_N - 1)

        return [padding + sentence for sentence in sentences] or [padding]

    def _style_scores(self, prepared):
        # Score prepared sentences with each style's model (sum of log probabilities)
        n = self.STYLE_N
        scores = [0] * len(self.models)
        for style_file, model in enumerate(self.models):
            for sentence in prepared:
                for i in range(len(sentence) - n + 1):
                    scores[style_file] += math.log(estimate_probability(
                        sentence[i + n - 1], sentence[i:i + n - 1], model))

        return scores

    def write_in_style_ngram(self, passage, k=10):
        """
        Takes a passage in, matches it with a style, and predicts the next
        word that will appear using a trigram model.

        Args:
            passage: A string that contains a passage

        Returns:
            top k words list<string>
            probabilities associated with each word list<float>
            index of "style" it originated from (e.g., 0 for 1st file) <int>
            probability associated with the style <float>
        """
        lines = [nltk.word_tokenize(line) for line in passage.split("\n") if line.strip()]
        prepared = self._prepare(lines)

        # Pick the style whose language model best fits the passage,
        # then softmax the scores to get its probability
        scores = self._style_scores(prepared)
        best_score = max(scores)
        exps = [math.exp(score - best_score) for score in scores]
        style_file = scores.index(best_score)
        probability_style = exps[style_file] / sum(exps)

        # Top k next words from the winning style, using the last (n - 1) tokens
        model = self.models[style_file]
        context_tokens = prepared[-1][-(self.STYLE_N - 1):]
        probabilities = {word: estimate_probability(word, context_tokens, model)
                         for word in next_words(model)}
        top_k = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]

        words = [word for word, _ in top_k]
        word_probabilities = [prob for _, prob in top_k]

        return words, word_probabilities, style_file, probability_style


# ### Example usage

# #### Twitter corpus

# In[15]:


get_ipython().run_cell_magic('time', '', '# Create an instance of your NGramModel here, using your training data\nspecial_tokens = SpecialTokens()\ncount_threshold = 10\ntrain_data_replaced, test_data_replaced, vocabulary = preprocess_data(\n    "data/en_US.twitter.txt", count_threshold, special_tokens\n)\n\npreprocess_data_test()\ncount_n_grams_test()\nestimate_probabilities_test()\n\n# n=2\nunigram_counts = count_n_grams(train_data_replaced, 1, special_tokens)\nbigram_counts = count_n_grams(train_data_replaced, 2, special_tokens)\n\nngram_model = NGramModel(unigram_counts, bigram_counts, vocabulary,\n                        special_tokens, k=1)\n')


# In[16]:


get_ipython().run_cell_magic('time', '', 'partial_sentence = "i love"  # Example partial sentence\npredicted_word = predict_next_word(partial_sentence, ngram_model)\nprint(f"The predicted next word for \'{partial_sentence}\' is: {predicted_word}")\n')


# #### Style Gram

# In[17]:


get_ipython().run_cell_magic('time', '', 'style_files = ["data/hemingway-edit.txt", "data/shakespeare-edit.txt", "data/dickinson-edit.txt"]\nstylegram = StyleGram(style_files)\n')


# In[18]:


def stylegram_accuracy(stylegram, lines_per_passage=5):
    correct, total = 0, 0
    for index, lines in enumerate(stylegram.test_data):
        for i in range(0, len(lines), lines_per_passage):
            scores = stylegram._style_scores(lines[i:i + lines_per_passage])
            correct += (scores.index(max(scores)) == index)
            total += 1
    return correct / total

# Spot checks: known passages and output format
tests = [
    # (passage, expected style index)
    ("The man drank his coffee and looked at the river. It was cold and he was tired", 0),
    ("Shall I compare thee to a summer's day? Thou art more lovely", 1),
    ("Because I could not stop for Death, He kindly stopped for me", 2),
]

for passage, expected in tests:
    words, probs, style, style_prob = stylegram.write_in_style_ngram(passage, k=10)

    assert style == expected, f"Wrong style for {passage!r}: got {style}, expected {expected}"
    assert len(words) == 10 and len(probs) == 10, "Should return k words and k probabilities"
    assert all(isinstance(w, str) for w in words), "Words should be strings"
    assert probs == sorted(probs, reverse=True), "Probabilities should be sorted high to low"
    assert 0.0 <= style_prob <= 1.0, "Style probability should be between 0 and 1"

    print(f"  {passage} → style {style} ({style_prob:.2f}), top 3: {words[:3]}")

# Accuracy on held-out passages
accuracy = stylegram_accuracy(stylegram)
print(f"  Held-out accuracy: {accuracy:.1%}")

print("\033[92m Successful test")


# In[ ]:




