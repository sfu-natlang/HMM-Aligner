# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of HMM word aligner, it requires IBM1 in order to
# function properly
#
import cython
import numpy as np
from collections import defaultdict
from loggers import logging
from models.cIBM1 import AlignmentModel as AlignerIBM1
from models.cHMMBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


@cython.boundscheck(False)
class AlignmentModel(Base):
    def __init__(self):
        self.modelName = "HMM"
        self.version = "0.4b"
        self.logger = logging.getLogger('HMM')
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.evaluate = evaluate
        self.fLex = self.eLex = self.fIndex = self.eIndex = None

        self.modelComponents = ["t", "pi", "a", "eLengthSet",
                                "fLex", "eLex", "fIndex", "eIndex"]
        Base.__init__(self)
        return

    def _beginningOfIteration(self, dataset, maxE, index):
        self.lenDataset = len(dataset)
        self.gammaEWord = np.zeros(len(self.eLex[index]))
        self.gammaBiword = [defaultdict(float)
                            for i in range(len(self.fLex[index]))]
        self.gammaSum_0 = np.zeros(maxE)
        return

    def EStepGamma(self, f, e, gamma, index):
        cdef int fLen = len(f)
        cdef int eLen = len(e)
        fWords = np.array([f[i][index] for i in range(fLen)])
        eWords = np.array([e[j][index] for j in range(eLen)])
        for i in range(fLen):
            for j in range(eLen):
                self.gammaBiword[fWords[i]][eWords[j]] += gamma[i][j]
        self.gammaSum_0[:eLen] += gamma[0]

        eDupli = (eWords[:, np.newaxis] == eWords).sum(axis=0)
        self.gammaEWord[eWords] += (gamma * eDupli).sum(axis=0)
        return

    def MStepDelta(self, maxE, index):
        # Update a
        for Len in self.eLengthSet:
            deltaSum = np.sum(self.delta[Len], axis=1) + 1e-37
            for prev_j in range(Len):
                self.a[Len][prev_j][:Len] =\
                    self.delta[Len][prev_j][:Len] / deltaSum[prev_j]

    def MStepGamma(self, maxE, index):
        # Update pi
        self.pi[:maxE] = self.gammaSum_0[:maxE] / self.lenDataset

        # Update t
        for i in range(len(self.fLex[index])):
            for j in self.gammaBiword[i]:
                self.t[i][j] = self.gammaBiword[i][j] / self.gammaEWord[j]
        return

    def endOfBaumWelch(self, index):
        # Smoothing for target sentences of unencountered length
        for targetLen in self.eLengthSet:
            a = self.a[targetLen]
            for prev_j in range(targetLen):
                for j in range(targetLen):
                    a[prev_j][j] *= 1 - self.p0H
        for targetLen in self.eLengthSet:
            a = self.a[targetLen]
            for prev_j in range(targetLen):
                for j in range(targetLen):
                    a[prev_j][prev_j + targetLen] = self.p0H
                    a[prev_j + targetLen][prev_j + targetLen] = self.p0H
                    a[prev_j + targetLen][j] = a[prev_j][j]
        return

    def train(self, dataset, iterations):
        dataset = self.initialiseLexikon(dataset)
        self.logger.info("Training IBM model 1")
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.sharedLexikon(self)
        alignerIBM1.initialiseBiwordCount(dataset)
        alignerIBM1.EM(dataset, iterations, 'IBM1')
        self.t = alignerIBM1.t
        self.logger.info("IBM model Trained")
        self.baumWelch(dataset, iterations=iterations)
        return
