# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of HMM word aligner, it requires IBM1 in order to
# function properly
#
import numpy as np
from collections import defaultdict
from loggers import logging
from models.cIBM1 import AlignmentModel as AlignerIBM1
from models.cModelBase import Task
from models.cHMMBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(Base):
    def __init__(self):
        self.modelName = "HMM"
        self.version = "0.3b"
        self.logger = logging.getLogger('HMM')
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.task = None
        self.evaluate = evaluate
        self.fLex = self.eLex = self.fIndex = self.eIndex = None

        self.modelComponents = ["t", "pi", "a", "eLengthSet",
                                "fLex", "eLex", "fIndex", "eIndex"]
        Base.__init__(self)
        return

    def _beginningOfIteration(self, dataset, maxE, index):
        self.lenDataset = len(dataset)
        self.gammaEWord = [0.0 for i in range(len(self.eLex[index]))]
        self.gammaBiword = [defaultdict(float)
                            for i in range(len(self.fLex[index]))]
        self.gammaSum_0 = np.zeros(maxE)
        return

    def _updateGamma(self, f, e, alpha, beta, alphaScale, index):
        fWords = [f[i][index] for i in range(len(f))]
        eWords = [e[j][index] for j in range(len(e))]
        gamma = ((alpha * beta).T / alphaScale).T
        for i in range(len(f)):
            for j in range(len(e)):
                self.gammaBiword[fWords[i]][eWords[j]] += gamma[i][j]
                self.gammaEWord[eWords[j]] += gamma[i][j]
        self.gammaSum_0[:len(e)] += gamma[0]
        return gamma

    def _updateEndOfIteration(self, maxE, index):
        self.logger.info("End of iteration")
        # Update a
        for Len in self.eLengthSet:
            deltaSum = np.sum(self.delta[Len], axis=1) + 1e-37
            for prev_j in range(Len):
                self.a[Len][prev_j][:Len] =\
                    self.delta[Len][prev_j][:Len] / deltaSum[prev_j]

        # Update pi
        self.pi[:maxE] = self.gammaSum_0[:maxE] / self.lenDataset

        # Update t
        for i in range(len(self.fLex[index])):
            for j in self.gammaBiword[i]:
                self.t[i][j] = self.gammaBiword[i][j] / self.gammaEWord[j]
        del self.gammaEWord
        del self.gammaBiword
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
        self.task = Task("Aligner", "HMMOI" + str(iterations))
        self.task.progress("Training IBM model 1")
        self.logger.info("Training IBM model 1")
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.sharedLexikon(self)
        alignerIBM1.initialiseBiwordCount(dataset)
        alignerIBM1.EM(dataset, iterations, 'IBM1')
        self.t = alignerIBM1.t
        self.task.progress("IBM model Trained")
        self.logger.info("IBM model Trained")
        self.baumWelch(dataset, iterations=iterations)
        self.task.progress("finalising")
        self.task = None
        return
