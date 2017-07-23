# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of the extended HMM word aligner as described in
# Toutanova's 2002 paper (combination of 5.1 and 5.3).
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
        self.modelName = "ExtendedHMM"
        self.version = "0.1b"
        self.tTags = []
        self.pStay = np.zeros(0)
        self.modelComponents += ["tTags"]
        self.modelComponents += ["pStay"]
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

    def _beginningOfIteration(self, dataset, maxE, index):
        HMM._beginningOfIteration(self, dataset, maxE, index)
        self.gammaBiTags = [defaultdict(float)
                            for i in range(len(self.fLex[1]))]
        self.gammaETags = [0.0 for i in range(len(self.eLex[1]))]
        return

    def EStepGamma(self, f, e, gamma, index):
        HMM.EStepGamma(self, f, e, gamma, index)
        for i in range(len(f)):
            for j in range(len(e)):
                self.gammaBiTags[f[i][1]][e[j][1]] += gamma[i][j]
                self.gammaETags[e[j][1]] += gamma[i][j]
        return

    def MStepGamma(self, maxE, index):
        HMM.MStepGamma(self, maxE, index)
        # Update tTags
        for i in range(len(self.fLex[1])):
            for j in self.gammaBiTags[i]:
                self.tTags[i][j] = self.gammaBiTags[i][j] / self.gammaETags[j]
        return

    def EStepDelta(self, f, e, xi):
        HMM.EStepDelta(self, f, e, xi)
        for j in range(len(e)):
            self.newPStay[e[j][1]] += self.delta[len(e)][j][j]

    def MStepDelta(self, maxE, index):
        HMM.MStepDelta(self, maxE, index)
        self.pStay = self.newPStay / np.sum(self.newPStay)
        return

    def aProbability(self, f, e):
        a = np.array(HMM.aProbability(self, f, e), copy=True)
        for i in range(len(e)):
            if i < len(self.eLex):
                a[:, i] *= (1 - self.pStay[e[i][1]])
                a[:, i, i] = self.pStay[e[i][1]]
        return a

    def tProbability(self, f, e, index=0):
        tmp = self.t
        t = HMM.tProbability(self, f, e, 0)
        self.t = self.tTags
        tTags = HMM.tProbability(self, f, e, 1)
        self.t = tmp
        return t * tTags

    def train(self, dataset, iterations):
        dataset = self.initialiseLexikon(dataset)
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.sharedLexikon(self)
        self.logger.info("Training IBM model 1 on FORM")
        alignerIBM1.initialiseBiwordCount(dataset, index=0)
        alignerIBM1.EM(dataset, iterations, 'IBM1', index=0)
        self.t, alignerIBM1.t = alignerIBM1.t, []
        self.logger.info("Training IBM model 1 on POSTAG")
        alignerIBM1.initialiseBiwordCount(dataset, index=1)
        alignerIBM1.EM(dataset, iterations, 'IBM1', index=1)
        self.tTags = alignerIBM1.t
        self.baumWelch(dataset, iterations=iterations)
        return
