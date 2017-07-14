# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of the extended HMM word aligner as described in
# Toutanova's 2002 paper (5.2). It adds in Tag Sequences for Jump Probabilities
#
import numpy as np
from collections import defaultdict
from loggers import logging
try:
    from models.cIBM1 import AlignmentModel as AlignerIBM1
    from models.cHMM import AlignmentModel as HMM
except ImportError:
    from models.IBM1 import AlignmentModel as AlignerIBM1
    from models.HMM import AlignmentModel as HMM
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(HMM):
    def __init__(self):
        HMM.__init__(self)
        self.modelName = "Toutanova2"
        self.version = "0.1b"
        return

    def initialValues(self, Len):
        self.a[:Len + 1, :, :Len, :Len].fill(1.0 / Len)
        self.pi[:Len].fill(1.0 / 2 / Len)
        return

    def initialiseParameter(self, maxE):
        self.a =\
            np.zeros((maxE + 1, len(self.fLex[1]) + 1, maxE * 2, maxE * 2))
        self.pi = np.zeros(maxE * 2)
        self.delta = np.zeros((maxE + 1, len(self.fLex[1]) + 1, maxE, maxE))
        return

    def aProbability(self, f, e):
        Len = len(e)
        if Len in self.eLengthSet:
            a = np.zeros((len(f), Len * 2, Len * 2))
            for i in range(len(f)):
                if f[i][1] < len(self.fLex[1]):
                    a[i] = self.a[Len][f[i][1]][:Len * 2, :Len * 2]
                else:
                    a[i] = self.a[Len][-1][:Len * 2, :Len * 2]
                a[i][np.where(a[i] == 0)] =\
                    self.a[Len][-1][np.where(a[i] == 0)]
            return a
        return np.tile(np.full((Len * 2, Len * 2), 1. / Len), (len(f), 1, 1))

    def EStepDelta(self, f, e, xi):
        fLen, eLen = len(f), len(e)
        c = np.zeros((fLen, eLen * 2))
        for i in range(fLen):
            for j in range(eLen):
                c[i][eLen - 1 - j:2 * eLen - 1 - j] += xi[i][j]
        for i in range(fLen):
            for j in range(eLen):
                self.delta[eLen][f[i][1]][j][:eLen] +=\
                    c[i][eLen - 1 - j:2 * eLen - 1 - j]

        c = np.zeros(eLen * 2)
        Xceta = np.sum(xi, axis=0)
        for j in range(eLen):
            c[eLen - 1 - j:2 * eLen - 1 - j] += Xceta[j]
        for j in range(eLen):
            self.delta[eLen][-1][j][:eLen] +=\
                c[eLen - 1 - j:2 * eLen - 1 - j]

    def MStepDelta(self, maxE, index):
        for Len in self.eLengthSet:
            for fTag in range(len(self.fLex[1]) + 1):
                deltaSum = np.sum(self.delta[Len][fTag], axis=1) + 1e-37
                for prev_j in range(Len):
                    self.a[Len][fTag][prev_j][:Len] =\
                        self.delta[Len][fTag][prev_j][:Len] / deltaSum[prev_j]
        return

    def endOfBaumWelch(self, index):
        for targetLen in self.eLengthSet:
            a = self.a[targetLen]
            for prev_j in range(targetLen):
                for j in range(targetLen):
                    a[:, prev_j, j] *= 1 - self.p0H
        for targetLen in self.eLengthSet:
            a = self.a[targetLen]
            for prev_j in range(targetLen):
                for j in range(targetLen):
                    a[:, prev_j, prev_j + targetLen] = self.p0H
                    a[:, prev_j + targetLen, prev_j + targetLen] = self.p0H
                    a[:, prev_j + targetLen, j] = a[:, prev_j, j]
        return
