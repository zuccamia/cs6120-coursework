#!/usr/bin/env python
# coding: utf-8

# # Assignment 1

# ## Setup & Preprocessing

# In[108]:


get_ipython().system('curl -k -O https://course.ccs.neu.edu/cs6120f26/data/shakespeare/shakespeare-edit.txt')


# In[137]:


import re
import string
import unicodedata
from collections import Counter


def read_vocabulary(filename: str) -> List[str]:
    """
    Reads in a given file specified by "filename" and processes it
    by removing punctuation, forcing lowercase, splits into
    individual words, and removes the numbers that might appear in
    the text.
    Args:
        filename: the name of the file to be processed
    Returns:
        A list of words in the order in which they appeared in the
    text.
    """
    counts = Counter()
    with open(filename, encoding="utf-8") as f:
        # stream line by line, keeping only 1 line + 8KB buffer in memory 
        for line in f:
            # Assume that Shakespearean texts are mostly ASCII-safe but for
            # some special legacy characters. Examine for any non-ASCII characters:
            for c in line:
                if not c.isascii():
                    print(f"Non-ASCII character found: {c!r}")
            # Examine for 
            # Normalize to ASCII for consistent word counting.
            # - Accent café -> cafe +, ligature ﬁ -> fi, full-width Ａ-> A
            line = unicodedata.normalize("NFKD", line)
            line.encode("ascii", "ignore").decode()
            # - Remove all punctuations, numerical digits and then lowercase all.
            # - EXCLUDE hyphens and apostrophes as they may be part of valid, 
            #   even if archaic words (e.g. to-morrow, thatch'd)
            to_remove = (string.punctuation + string.digits).replace("'", "").replace("-", "")
            line = line.translate(str.maketrans("", "", to_remove))
            # - Remove stand-alone hyphens that don't actually make up a word.
            line = re.sub(r"(?<!\w)-|-(?!\w)", "", line)
            counts.update(line.lower().split())

    return [word for word, _ in counts.most_common()]


# In[138]:


word_list = read_vocabulary("shakespeare-edit.txt")


# #### Test

# In[139]:


assert len(word_list) > 0, "word list was not populated"
print(f"Total unique word count: {len(word_list)}")
assert word_list[0] == "the", f"expected \"the\", got {word_list[0]}"
assert word_list[1] == "and", f"expected \"and\", got {word_list[1]}"
assert word_list[2] == "i", f"expected \"i\", got {word_list[2]}"
print("Tests passed")


# ## Autocomplete

# In[140]:


from operator import itemgetter


# Trie is the most speed-efficient data structure for prefix lookups for a large vocabulary size
# in exchange for memory overhead.
# List + binary search isn't too bad for this Shakespear corpus's vocabulary size, but for the sake
# of practicing with Trie as a new data structure for me, and as the base of the actual data structure
# often used in production for prefix lookups, I'm using Trie as the DS of choice here.
class Trie:
    def __init__(self):
        """
        Initiate an empty Trie node, usually as the root of a Trie tree.
        Attributes:
            children: dictionary that stores child pair(s) of { <following_character>: <Trie node> }
            order: relative frequency, with 0 being the highest. 
                   Also a marker for end-of-word when order is not None (0 is a valid index but falsy in Python).
                   Only complete word gets assigned its index in the most common words list.
        """
        self.children = {}
        self.order = None

    def insert(self, word: str, idx: int) -> None:
        """
        Index a word string and its order of frequency to the Trie tree (i.e. "self") by:
            From the root of the Trie tree ("self"), for each char in the given word:
            check if the char exists in the current node's children.
            If yes, advance the current node to the node of the children[char].
            If not, create a new key of that char with a new empty Trie node children[char] = Trie()
            and advance current node to that new node.
            Assign the given frequency index of the word at the end after indexing all chars.
        Args:
            word: the word string to be indexed/inserted.
            idx: the relative frequency of the word in a predefined ordered list.
        """
        node = self
        for c in word:
            node = node.children.setdefault(c, Trie())
        node.order = idx

    def autocomplete(self, prefix: str) -> List[str]:
        """
        From a given string specified by "prefix", search through the Trie tree (i.e. "self") by
            1. From the root of the Trie tree ("self"), starting from the 1st char in the prefix,
            check if the 1st char exists in the root's children. 
            If not, that means there is no word starting with the prefix in the Trie, return early.
            If yes, advance the current root to the node of the children[char], and move to the next char.
            As the iteration completes, the root is now the Trie node of the children of the last char
            of the prefix.
            2. Initiate a list to store the autocomplete suggestions, consisting of tuples of (word, order).
            3. From the last char of the prefix, explore the Trie tree depth-first the different paths until
            there is no more matching chars in the Trie tree (node has no children).
            At any char, if a word/path is complete, add it to the suggestions list.
            Update the path with the current char key and keep exploring children nodes of the current char.
            4. Sort the suggestions list by the orders of the words, 0 being the highest/most common.
            5. Return only the ordered words as a list.
        Args:
            prefix: many words may start with this string
        Returns:
            A list of words that start with the "prefix" string in the order of most to least common.
        """
        root = self
        for c in prefix:
            if c not in root.children:
                return []
            root = root.children[c]

        suggestions = []

        def dfs(node: Trie, path: str) -> None:
            if node.order is not None:
                suggestions.append((prefix + path, node.order))
            for c, child in node.children.items():
                dfs(child, path + c)

        dfs(root, "")
        return [word for word, order in sorted(suggestions, key=itemgetter(1))]

    def display(self, indent: int = 0) -> None:
        for c, child in self.children.items():
            print(" "*indent + c + (f"({child.order})" if child.order is not None else ""))
            child.display(indent + 2)


def process_data(word_list: List[str]) -> Trie:
    """
    Builds Trie data structure from a given list of words sorted in the order of 
    the frequency (probability) of the words. 
    Word at index 0 in the list is the most common word.
    Args:
        words: A list of words.
    Returns:
        A Trie tree
    """
    root = Trie()
    for i, word in enumerate(word_list):
        root.insert(word, i)

    return root


# In[141]:


def autocomplete_word(prefix: str, trie: Trie) -> List[str]:
    """
    Returns a list of words starting with the given prefix. This list
    is sorted in order of the frequency (probability) of the words.
    Args:
        prefix: The prefix to search for.
        trie: A Trie tree index of words storing the frequency order for each
              complete word.
    Returns:
        A list of ten most-common words starting with the prefix.
    """
    return trie.autocomplete(prefix)


# #### Test

# In[142]:


truncated = word_list[:10]
assert truncated == ['the', 'and', 'i', 'to', 'of', 'a', 'you', 'my', 'in', 'that'], f"expected \"['the', 'and', 'i', 'to', 'of', 'a', 'my', 'in', 'you', 'that']\", got {truncated}"

# process_data() returning the root of a Trie as the DS/model of choice
trie = process_data(truncated)
trie.display()

# autocomplete_word()
suggestions_th = autocomplete_word("th", trie)
assert suggestions_th == ['the', 'that'], f"expected \"['the', 'that']\", got {suggestions_th}"
suggestions_re = autocomplete_word("re", trie)
assert suggestions_re == [], f"expected no suggestions, got {suggestions_re}"

print("Tests passed")


# In[143]:


model_or_data_structure = process_data(word_list)


# ## Text Box with Autocompletion

# In[144]:


import ipywidgets as widgets
from IPython.display import display


def on_value_change(change):
    prefix = change["new"]
    suggestions = autocomplete_word(prefix, model_or_data_structure)
    with output:
        output.clear_output()
        if suggestions:
            print("Suggestions:")
            for word in suggestions:
                print(word)
        else:
            print("No suggestions found.")

text = widgets.Text()
output = widgets.Output()
display(text, output)
text.observe(on_value_change, names="value")


# In[ ]:




