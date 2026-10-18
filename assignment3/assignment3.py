#!/usr/bin/env python
# coding: utf-8

# # Assignment 3 - Autocorrect

# ## Setup & Processing

# In[117]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import re
import string
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd


# ## Q1 Process Data

# In[118]:


def process_data(file_name):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        wordprobs: a dictionary where keys are all the processed lowercase words and the 
             values are the frequency that it occurs in the corpus (text file you read).
    """
    with open(file_name, encoding="utf-8") as f:
        data = f.read()
    lower_case_data = data.lower()
    words = re.findall(r'\w+', lower_case_data)
    freqs = Counter(words)
    total = sum(freqs.values())
    wordprobs = {word: freq/total for word, freq in freqs.items()}

    return wordprobs

def process_data_test():

    probs = process_data('./shakespeare-7k.txt')
    assert np.allclose(probs['thee'], 0.004476442720185026), \
                      f"expected \"thee\" frequency of 0.004476442720185026, got {probs['thee']}"
    assert len(probs) == 6116, "vocabulary should have total of 6116 words.\n\t" + \
        f"length of vocabulary in your implementation is {len(probs)}"

    # Check maximum frequency word
    max_word = ""; max_freq = 0.0
    for word in probs:
      assert word == word.lower(), "all words must be lowercase.\n\t" + \
          f"word \"{word}\" is not lowercase"
      if probs[word] > max_freq:
        max_freq = probs[word]
        max_word = word
    assert max_word == 'the', "\"the\" should be the most frequent word\n\t" + \
        f"your maximum word is \"{max_word}\""
    assert np.allclose(max_freq, 0.028444063117842356)

    print("process_data: \033[1;32mtests OK.\033[0m")


# In[119]:


process_data_test()


# ## Q2 N-gram most likely next words

# In[120]:


def probable_substitutes(word, probs, maxret = 10):
    """
    Determine the most probable words for a misspelled string that are TWO edits
    away. The edits that are possible are:

        * delete_letter: removing a character
        * switch_letters: switching two adjacent characters
        * replace_letter: replacing one character by another different one
        * insert_letter: inserting a character

    There may be fewer but no more than maxret words.

    Input:
        word - The misspelled word
        probs - A dictionary of word --> prob
        maxret - Maximum number of words to return
    Returns:
        Tuples of the words and their probabilities, ordered by highest frequency.
        [(word1, prob1), ... ]
    """
    def alphabet():
        return "abcdefghijklmnopqrstuvwxyz"

    def delete_letter(word):
        """
        Remove any one letter from the word (complexity: O(n))
        """
        return [word[:i] + word[i+1:] for i in range(len(word))]

    def switch_letters(word):
        """
        Swap any 2 adjacent letters in a word (complexity: O(n - 1))
        """
        return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word)-1)]

    def replace_letter(word):
        """
        Replace any one letter by another different one in the English alphabet (complexity: O(25*n)).
        All letters in word are lowercase.
        """
        subs = []
        for i in range(len(word)):
            for c in alphabet():
                if word[i] != c:
                    subs.append(word[:i] + c + word[i+1:])
        return subs

    def insert_letter(word):
        """
        Insert any letter of the English alphabet into any letter position in the word (complexity: O(26(n + 1)))
        """
        subs = []
        for i in range(len(word)+1):
            for c in alphabet():
                subs.append(word[:i] + c + word[i:])
        return subs

    def edit_one_letter(word):
        """
        Apply any of the possible edits:

        * delete_letter: removing a character
        * switch_letters: switching two adjacent characters
        * replace_letter: replacing one character by another different one
        * insert_letter: inserting a character

        Return a set of edited words
        """
        edits = set()
        edits.update(delete_letter(word))
        edits.update(switch_letters(word))
        edits.update(replace_letter(word))
        edits.update(insert_letter(word))

        return edits

    edits = set()
    for w in edit_one_letter(word):
        edits.update(edit_one_letter(w))
    subs = edits & probs.keys()
    ranked = sorted(subs, key=lambda w: probs[w], reverse=True)
    return [(w, probs[w]) for w in ranked][:maxret]

def probable_substitutes_test():
    probs = dict()
    probs['raid'] = 0.25
    probs['rain'] = 0.1
    probs['raise'] = 0.1
    probs['void'] = 0.15
    probs['roid'] = 0.20
    probs['rein'] = 0.2

    # Check delete one add one
    probable_substitutes_result = probable_substitutes('braid', probs, maxret = 5)   
    probable_substitutes_expected = [('raid', 0.25), ('roid', 0.2), ('rain', 0.1)]
    assert probable_substitutes_result == probable_substitutes_expected, \
      str(probable_substitutes_expected) + "\n\tReceived" + str(probable_substitutes_result)

    # Check insert two
    probable_substitutes_result = probable_substitutes('id', probs, maxret = 5)   
    probable_substitutes_expected = [('raid', 0.25), ('roid', 0.2), ('void', 0.15)]
    assert probable_substitutes_result == probable_substitutes_expected, \
      str(probable_substitutes_expected) + "\n\tReceived" + str(probable_substitutes_result)

    print("probable_substitutes: \033[1;32mtests OK.\033[0m")


# In[121]:


probable_substitutes_test()


# ## Q3 The Minimum Edit Distance

# In[122]:


def min_edit_distance(source, target, ins_cost = 1, 
                      del_cost = 1, rep_cost = 2):
    '''
    Input:
        source: starting string
        target: ending string
        ins_cost: integer representing insert cost
        del_cost: integer representing delete cost
        rep_cost: integer representing replace cost
    Output:
        D: matrix of size (len(source)+1 , len(target)+1) 
           with minimum edit distances
        med: the minimum edit distance required to convert
             source to target
    '''
    m = len(source)
    n = len(target)
    # Rows: letters in source
    # Columns: letters in target
    D = [[0] * (n+1) for _ in range(m+1)]

    # Base case: no overlapping letters
    for i in range(1, m+1):
        D[i][0] = i * del_cost # delete every letter of the source
    for j in range(1, n+1):
        D[0][j] = j * ins_cost # insert every letter of the target

    # Calculate the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            r_cost = 0 if source[i-1] == target[j-1] else rep_cost
            D[i][j] = min(D[i-1][j] + del_cost, D[i][j-1] + ins_cost, D[i-1][j-1] + r_cost)
    return D, D[m][n]

def min_edit_distance_test():

    matrix, mindist = min_edit_distance("hello", "help", ins_cost = 1, del_cost = 1, rep_cost = 2)
    assert mindist == 3, "Expected distance of 3, but received: " + str(mindist)   

    matrix, mindist = min_edit_distance("hello", "jello", ins_cost = 1, del_cost = 1, rep_cost = 2)
    assert mindist == 2, "Expected distance of 2, but received: " + str(mindist)

    matrix, mindist = min_edit_distance("ello", "jello", ins_cost = 1, del_cost = 1, rep_cost = 2)
    assert mindist == 1, "Expected distance of 1, but received: " + str(mindist)

    print("min_edit_distance: \033[1;32mtests OK.\033[0m")


# In[84]:


min_edit_distance_test()


# ### Full test

# In[85]:


if __name__ == '__main__':

    print("Running unit tests ... ")
    process_data_test()
    probable_substitutes_test()
    min_edit_distance_test()

    print("Processing data, assuming that shakespeare-7k.txt is in working directory...")
    probs = process_data('./shakespeare-7k.txt')

    print("\n\nLet's try autocorrect. \"(Q)/(q)uit\" to end.")
    entry = input("Enter any word:")
    entry = entry.lower()
    while not entry == "q" and not entry == 'quit':
        print(probable_substitutes(entry, probs, maxret = 5))
        entry = input("Enter any word:")
        entry = entry.lower()

    print("\n\nLet's try min edit distance.")
    while not entry == 'n' and not entry == 'no':
        source = input("Enter any source word:")
        source = source.lower()
        target = input("Enter any target word:")
        target = target.lower()
        D, d = min_edit_distance(source, target)
        print(source, " and ", target, " are ", d, " apart")
        entry = input("Continue? (Y/y)es / (N/n)o:")


# ## Q4 Tokenization

# In[123]:


def merge_pair(syms, best_pair):
    """
    Input:
        syms: a tuple of symbols that make up a word
        best_pair: a tuple of a pair of symbols found to be the most common in a vocabulary
    Output:
        A tuple of symbols that make up a word but with every occurence matching the pair symbol merged as one single symbol
    E.g. merge_pair(('l','o','w','</w>', ('l','o')) returns ('lo', 'w', '</w>')
    """
    new_syms = []
    i = 0
    while i < len(syms):
        if i < (len(syms)-1) and (syms[i], syms[i+1]) == best_pair:
            sym = syms[i] + syms[i+1]
            new_syms.append(sym)
            i += 2
        else:
            new_syms.append(syms[i])
            i += 1
    return tuple(new_syms)

def bpe_learn_merges(corpus, num_merges=5):
    """
    Input:
        corpus: string containing space-separated words
        num_merges: integer specifying how many merge operations to learn
    Output:
        merges: list of tuples (pair_symbol_1, pair_symbol_2) ordered by learning
        sequenvocab: dict of final vocabulary tokens and their frequencies
    """
    # Step 1: Build a dict of tuple of characters in a word mapping to the word's frequency
    freqs = Counter(corpus.lower().split())
    vocab = {tuple(list(word) + ['</w>']): freq for word, freq in freqs.items()}

    merges = []
    for _ in range(num_merges):
    # Step 2: Count frequency of adjacent symbol pairs from the vocab (frequency of each symbol pair == frequency of their origin word)
        pair_freqs = Counter()
        for syms, freq in vocab.items():
            for i in range(len(syms) - 1):
                pair_freqs[(syms[i], syms[i+1])] += freq

    # Step 3: Pick the most frequent/common symbol pair, tie-break lexicographically
        best_pair = min(pair_freqs.items(), key=lambda item: (-item[1], item[0]))[0]
        merges.append(best_pair)

    # Step 4: Merge the best pair into a new symbol. Rebuild vocab with the chosen pair merged.
        new_vocab = {}
        for syms, freq in vocab.items():
            new_syms = merge_pair(syms, best_pair)
            new_vocab[new_syms] = new_vocab.get(new_syms, 0) + freq
        vocab = new_vocab

    # Step 5: Flatten new (merged) vocab into tokens and their frequency
    token_freqs = Counter()
    for syms, freq in vocab.items():
        for sym in syms:
            token_freqs[sym] += freq

    sequenvocab = dict(token_freqs)

    return merges, sequenvocab


# In[124]:


corpus = "low low low lower lower newest newest widest"
bpe_learn_merges(corpus, num_merges=3)


# In[ ]:




