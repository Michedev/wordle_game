import numpy as np
from collections import Counter
from itertools import product
from string import ascii_lowercase

cdef:
    int RESULT_CORRECT = 1
    int RESULT_IN_WORD = 2
    int RESULT_INVALID = 3

    class Constraint:
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

    class LetterDistribution:
        cdef:
            dict p_green
            dict p_yellow
            dict p_gray
            dict entropy_letters

        def __cinit__(self, p_green, p_yellow, p_gray):
            self.p_green = p_green
            self.p_yellow = p_yellow
            self.p_gray = p_gray
            self.entropy_letters = dict()
            for l in p_green:
                self.entropy_letters[l] = [0,0,0,0,0]
                for i in range(5):
                    l_distribution = [p_green[l][i], p_yellow[l][i], p_gray[l][i]]
                    l_distribution = np.array(l_distribution)
                    l_distribution = l_distribution / l_distribution.sum()
                    l_entropy = - (l_distribution * np.log2(l_distribution + 1e-5)).sum()
                    self.entropy_letters[l][i] = l_entropy


cpdef float match_probability(list constraints, list words, dict words_prob):
   cdef float num_total_words = sum(words_prob.values())
   for c in constraints:
    words = [w for w in words if c.match(w)]
   p_words = [words_prob[w] for w in words]
   return sum(p_words) / num_total_words


def compute_constraints_entropy(word: str, word_list: list, words_prob: dict):
    """
    Compute probability distribution of all possible constrains and then compute entropy
    Warning: exponential complexity!! Implement with cython and test before run on all words
    """
    distribution = []
    for w_constraints in product([3,2,1], repeat=5):
        if Counter(w_constraints) == {1:4,2:1}:
            continue
        constraints = [Constraint(w_constraints[i], i, word[i]) for i in range(5)]
        p = match_probability(constraints, word_list, words_prob)
        distribution.append(p)
    distribution = np.array(distribution)
    distribution = distribution[distribution > 0.0]
    entropy = - (distribution * np.log2(distribution)).sum()
    return entropy

cpdef LetterDistribution compute_letter_entropy_distribution(list word_list):
    cdef dict p_green = {ascii_lowercase[i]: [0, 0, 0, 0, 0] for i in range(26)}
    cdef dict p_yellow = {ascii_lowercase[i]: [0, 0, 0, 0, 0] for i in range(26)}
    cdef dict p_gray = {ascii_lowercase[i]: [0, 0, 0, 0, 0] for i in range(26)}
    cdef float p_gray_l
    for w in word_list:
        for i in range(5):
            p_green[w[i]][i] += 1
    for l in p_yellow:
        for i in range(5):
            for j in range(5):
                if i != j:
                    p_yellow[l][i] += p_green[l][j]
    for l in p_green:
        for i in range(5):
            p_gray[l][i] = len(word_list) - p_green[l][i] - p_yellow[l][i]
    return LetterDistribution(p_green, p_yellow, p_gray)

def compute_words_entropy_by_letters(list word_list, list letter_word_list):
    cdef LetterDistribution distr = compute_letter_entropy_distribution(letter_word_list)
    cdef dict result = dict()
    for w in word_list:
        result[w] = 0
        seen_letters = set()
        for i, l in enumerate(w):
            if l in seen_letters:
                continue
            seen_letters.add(l)
            result[w] += distr.entropy_letters[l][i]
        if np.isnan(result[w]) or np.isinf(result[w]):
            result[w] = 0
    return result



