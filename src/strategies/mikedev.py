from random import sample

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
import pyximport
pyximport.install()
from src.util.entropy import compute_constraints_entropy, compute_words_entropy_by_letters
from src.wordle import GuessResult, Result, Wordle
from path import Path


class Constraint:

    def __init__(self, letter: str, type: Result, index: int):
        self.letter = letter
        self.type = type
        self.index = index

    def match(self, word: str):
        if self.type.value == Result.CORRECT.value:
            return word[self.index] == self.letter
        elif self.type.value == Result.IN_WORD.value:
            return word[self.index] != self.letter and self.letter in word
        elif self.type.value == Result.INVALID.value:  # INVALID
            return self.letter not in word
        else:
            raise Exception()

    def __str__(self):
        return f"({self.letter = }, {self.type = }, {self.index = })"

    def __repr__(self): return str(self)

    __slots__ = ('letter', 'type', 'index')


class MikedevStrategy(BaseStrategy):

    def __init__(self, game: Wordle):
        super().__init__(game)
        self.constraints = []
        self.root_path = Path(__file__).parent.parent.parent
        with open(self.root_path / 'bag_of_words.txt') as f:
            self.words = f.read().split('\n')
        # import sqlite3
        # con = sqlite3.connect(self.root_path / 'word_entropy.db')
        # self.words_entropy = pd.read_sql_query("select word, entropy from word_info", con).set_index('word')
        # self.words_entropy = self.words_entropy.to_dict()['entropy']
        self.words_prob = pd.read_csv(self.root_path / 'unigram_freq_wordle.csv', usecols=['word', 'probability']).set_index('word')
        self.words_prob = self.words_prob.to_dict()['probability']
        self.words_entropy = compute_words_entropy_by_letters(self.words, self.words)
        self.all_words = self.words[:]
        print('================ init new game =========================')
        self.round = 0
        print(game._secret_word)

    def get_guess(self) -> str:
        """Returns the next guess to try."""
        if self.round > 0:
            self._update_entropy_()
        self.round += 1
        guessed_word = max(self.words, key=lambda w: self.words_entropy[w])
        print(f"{guessed_word = }")
        return guessed_word

    def make_guess(self, guess: str) -> GuessResult:
        """Makes the guess."""
        guess_result = super().make_guess(guess)
        new_constraints = []
        for i in range(len(guess_result.result)):
            constraint = Constraint(guess_result.guess[i], guess_result.result[i], i)
            new_constraints.append(constraint)
            self.constraints.append(constraint)
        while tmp_constrains := self.fix_single_overlap(new_constraints):
            new_constraints = tmp_constrains
        print('\n'.join(map(str, new_constraints)))
        for c in new_constraints:
            self.words = filter(c.match, self.words)
        self.words = list(self.words)

        print(f"{len(self.words) = }")
        print(f'{self.words[:5] = }')
        return guess_result

    def fix_single_overlap(self, new_constraints):
        for i in range(len(new_constraints)):
            for j in range(len(new_constraints)):
                if i != j and new_constraints[i].letter == new_constraints[j].letter and new_constraints[i].type.value != new_constraints[j].type.value:
                    copy_constraints = list(new_constraints)
                    new_constraint = min([copy_constraints[i], copy_constraints[j]], key=lambda c: c.type.value)
                    a, b = copy_constraints[i], copy_constraints[j]
                    copy_constraints.remove(a)
                    copy_constraints.remove(b)
                    copy_constraints.append(new_constraint)
                    return copy_constraints
        return False

    def _update_entropy_(self):
        self.words_entropy = compute_words_entropy_by_letters(self.words, self.words)
        # for word in self.words:
        #     self.words_entropy[word] = compute_constraints_entropy(word, self.words if len(self.words) < 400 else sample(self.words, 400), self.words_prob)
