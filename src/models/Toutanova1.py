# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of the extended HMM word aligner as described in
# Toutanova's 2002 paper (5.1). It adds in POS Tags for translation probability
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
        self.modelName = "Toutanova1"
        self.version = "0.1b"
        self.tTags = []
        self.modelComponents += ["tTags"]
        return

    def _beginningOfIteration(self, dataset, maxE, index):
        HMM._beginningOfIteration(self, dataset, maxE, index)
        self.gammaBiTags = [defaultdict(float)
                            for i in range(len(self.fLex[1]))]
        self.gammaETags = [0.0 for i in range(len(self.eLex[1]))]
        return

    def _updateGamma(self, f, e, alpha, beta, alphaScale, index):
        gamma = HMM._updateGamma(self, f, e, alpha, beta, alphaScale, index)
        for i in range(len(f)):
            for j in range(len(e)):
                self.gammaBiTags[f[i][1]][e[j][1]] += gamma[i][j]
                self.gammaETags[e[j][1]] += gamma[i][j]
        return gamma

    def _updateEndOfIteration(self, maxE, index):
        HMM._updateEndOfIteration(self, maxE, index)
        # Update tTags
        for i in range(len(self.fLex[1])):
            for j in self.gammaBiTags[i]:
                self.tTags[i][j] = self.gammaBiTags[i][j] / self.gammaETags[j]
        del self.gammaETags
        del self.gammaBiTags
        return

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
