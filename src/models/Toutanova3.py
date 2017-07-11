# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of the extended HMM word aligner as described in
# Toutanova's 2002 paper (5.2). It adds in the Modeling Fertility.
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
        self.modelName = "Toutanova3"
        self.version = "0.1b"
        self.pStay = np.zeros(0)
        return

    def initialiseParameter(self, maxE):
        HMM.initialiseParameter(self, maxE)
        self.pStay = np.zeros(len(self.eLex[1]))
        self.newPStay = np.zeros(len(self.eLex[1]))
        return

    def initialValues(self, Len):
        HMM.initialValues(self, Len)
        self.pStay.fill(1.0 / Len)
        return

    def _updateDelta(self, f, e, alpha, beta, alphaScale, tSmall, a):
        HMM._updateDelta(self, f, e, alpha, beta, alphaScale, tSmall, a)
        for j in range(len(e)):
            self.newPStay[e[j][1]] += self.delta[len(e)][j][j]

    def _updateEndOfIteration(self, maxE, index):
        HMM._updateEndOfIteration(self, maxE, index)
        self.pStay = self.newPStay / np.sum(self.newPStay)
        return

    def aProbability(self, f, e):
        a = np.array(HMM.aProbability(self, f, e), copy=True)
        for i in range(len(e)):
            if i < len(self.eLex):
                a[i] *= (1 - self.pStay[e[i][1]])
                a[i][i] = self.pStay[e[i][1]]
        return a
