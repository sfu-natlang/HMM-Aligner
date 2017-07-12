# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of the extended HMM word aligner as described in
# Toutanova's 2002 paper (5.4). It adds in the translation model for null
#
import numpy as np
from collections import defaultdict
from loggers import logging
from models.IBM1 import AlignmentModel as AlignerIBM1
from models.HMM import AlignmentModel as HMM
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(HMM):
    def __init__(self):
        HMM.__init__(self)
        self.modelName = "Toutanova4"
        self.version = "0.1b"
        self.pFNull = []
        self.pFNull2 = []
        self.modelComponents += ["pFNull"]
        return

    def trainPFNull(self, dataset):
        self.pFNull = [0.0 for i in range(len(self.fLex[0]))]
        self.pFNull2 = [defaultdict(float) for i in range(len(self.fLex[0]))]
        total = [0 for i in range(len(self.fLex[0]))]
        total2 = [defaultdict(int) for i in range(len(self.fLex[0]))]

        for (f, e, alignment) in dataset:
            fAlign = [-1 for i in range(len(f))]
            for align in alignment:
                f_i, e_j = align[:2]
                fAlign[f_i - 1] = e_j - 1
            for i in range(len(f) - 1):
                total[f[i][0]] += 1
                total2[f[i][0]][f[i + 1][0]] += 1
                if fAlign[i] == 0:
                    self.pFNull[f[i][0]] += 1
                    self.pFNull2[f[i][0]][f[i + 1][0]] += 1
            i = len(f) - 1
            total[f[i][0]] += 1
            if fAlign[i] == 0:
                self.pFNull[f[i][0]] += 1

        for i in range(len(self.pFNull)):
            self.pFNull[i] /= total[i]
            for j in total2[i]:
                self.pFNull2[i][j] /= total2[i][j]
        return

    def fNullProb(self, f):
        lambd = 0.5
        result = np.zeros(len(f))
        for i in range(len(f)):
            if f[i] >= len(self.fLex[0]):
                result[i] = 0
            elif i + 1 >= len(f) or f[i + 1] not in self.pFNull2[f[i]]:
                result[i] = self.pFNull[f[i]] * (1 - lambd)
            else:
                result[i] = self.pFNull[f[i]] * (1 - lambd) +\
                    self.pFNull2[f[i]][f[i + 1]] * lambd
        return result

    def tProbability(self, f, e, index=0):
        t = np.zeros((len(f), len(e)))
        NullValues = self.fNullProb([f[i][0] for i in range(len(f))])
        for j in range(len(e)):
            if e[j][index] == 424242424243:
                for i in range(len(f)):
                    t[:, j] = NullValues
                continue
            if e[j][index] >= len(self.eLex[index]):
                continue
            for i in range(len(f)):
                if f[i][index] < len(self.t) and \
                        e[j][index] in self.t[f[i][index]]:
                    t[i][j] = self.t[f[i][index]][e[j][index]]
        t[t == 0] = 0.000006123586217
        return t

    def baumWelch(self, dataset, iterations=5, index=0):
        self.logger.info("Training NULL")
        self.trainPFNull(dataset)
        HMM.baumWelch(self, dataset, iterations, index)
