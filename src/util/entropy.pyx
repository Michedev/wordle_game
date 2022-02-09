import numpy as np
from collections import Counter
from itertools import product

cdef:
    int RESULT_CORRECT = 1
    int RESULT_IN_WORD = 2
    int RESULT_INVALID = 3


cdef class Constraint:
    cdef public:
        int type_
        int index
        str letter

    def __cinit__(self, int type_, int index, str letter):
        self.type_ = type_
        self.index = index
        self.letter = letter

    cpdef bint match(self, str word):
        if self.type_ == RESULT_CORRECT:
            return word[self.index] == self.letter
        elif self.type_ == RESULT_IN_WORD:
            return word[self.index] != self.letter and self.letter in word
        elif self.type_ == RESULT_INVALID:  # INVALID
            return self.letter not in word
        else:
            print(self.type_)
            raise Exception()


cpdef float match_probability(list constraints, list words):
   cdef float num_total_words = len(words)
   for c in constraints:
       words = [w for w in words if c.match(w)]
   return len(words) / num_total_words


def compute_constraints_entropy(word: str, word_list: list):
    """
    Compute probability distribution of all possible constrains and then compute entropy
    Warning: exponential complexity!! Implement with cython and test before run on all words
    """
    distribution = []
    for w_constraints in product([3,2,1], repeat=5):
        if Counter(w_constraints) == {1:4,2:1}:
            continue
        constraints = [Constraint(w_constraints[i], i, word[i]) for i in range(5)]
        p = match_probability(constraints, word_list)
        distribution.append(p)
    distribution = np.array(distribution)
    distribution = distribution[distribution > 0.0]
    entropy = - (distribution * np.log2(distribution)).sum()
    return entropy
