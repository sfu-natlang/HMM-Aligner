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
        return

    def _beginningOfIteration(self, dataset, maxE, index):
        HMM._beginningOfIteration(self, dataset, maxE, index)
        self.gammaBiTags = [defaultdict(float)
                            for i in range(len(self.fLex[1]))]
        self.gammaETags = [0.0 for i in range(len(self.eLex[1]))]
        return

    def gamma(self, f, e, alpha, beta, alphaScale, index):
        gamma = HMM.gamma(self, f, e, alpha, beta, alphaScale, index)
        fTags = [f[i][1] for i in range(len(f))]
        eTags = [e[j][1] for j in range(len(e))]
        for i in range(len(f)):
            for j in range(len(e)):
                self.gammaBiTags[fTags[i]][eTags[j]] += gamma[i][j]
                self.gammaETags[eTags[j]] += gamma[i][j]
        return gamma

    def _updateEndOfIteration(self, maxE, delta, index):
        HMM._updateEndOfIteration(self, maxE, delta, index)
        # Update tTags
        for i in range(len(self.fLex[1])):
            for j in self.gammaBiTags[i]:
                self.tTags[i][j] = self.gammaBiTags[i][j] / self.gammaETags[j]
        del self.gammaETags
        del self.gammaBiTags
        return

    def tProbability(self, f, e, index=0):
        t = np.zeros((len(f), len(e)))
        for j in range(len(e)):
            if e[j][0] == 424242424243:
                t[:, j].fill(self.nullEmissionProb)
                continue
            if e[j][0] >= len(self.eLex[0]):
                continue
            for i in range(len(f)):
                if f[i][0] < len(self.t) and \
                        e[j][0] in self.t[f[i][0]]:
                    t[i][j] = self.t[f[i][0]][e[j][0]]
        t[t == 0] = 0.000006123586217

        tTags = np.zeros((len(f), len(e)))
        for j in range(len(e)):
            if e[j][1] == 424242424243:
                tTags[:, j].fill(self.nullEmissionProb)
                continue
            if e[j][1] >= len(self.eLex[1]):
                continue
            for i in range(len(f)):
                if f[i][1] < len(self.tTags) and \
                        e[j][1] in self.tTags[f[i][1]]:
                    tTags[i][j] = self.tTags[f[i][1]][e[j][1]]
        tTags[tTags == 0] = 0.000006123586217
        return t * tTags

    def train(self, dataset, iterations):
        dataset = self.initialiseLexikon(dataset)
        self.logger.info("Training IBM model 1 on FORM")
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.sharedLexikon(self)
        alignerIBM1.initialiseBiwordCount(dataset, index=0)
        alignerIBM1.EM(dataset, iterations, 'IBM1', index=0)
        self.t = alignerIBM1.t
        self.logger.info("IBM model Trained")
        self.logger.info("Training IBM model 1 on POSTAG")
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.sharedLexikon(self)
        alignerIBM1.initialiseBiwordCount(dataset, index=1)
        alignerIBM1.EM(dataset, iterations, 'IBM1', index=1)
        self.tTags = alignerIBM1.t
        self.logger.info("IBM model Trained")
        self.baumWelch(dataset, iterations=iterations)
        return
