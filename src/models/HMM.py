# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of HMM word aligner, it requires IBM1 in order to
# function properly
#
from collections import defaultdict
from loggers import logging
from models.IBM1 import AlignmentModel as AlignerIBM1
from models.modelBase import Task
from models.HMMBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(Base):
    def __init__(self):
        self.modelName = "HMM"
        self.version = "0.1b"
        self.logger = logging.getLogger('HMM')
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.task = None
        self.evaluate = evaluate

        self.modelComponents = ["t", "pi", "a", "eLengthSet"]
        Base.__init__(self)
        return

    def _beginningOfIteration(self, dataset):
        self.lenDataset = len(dataset)
        return

    def _updateGamma(self, f, e, gamma, alpha, beta, alphaScale):
        for i in range(len(f)):
            for j in range(len(e)):
                gamma[i][j] = alpha[i][j] * beta[i][j] / alphaScale[i]

    def _updateEndOfIteration(self, maxE, delta, gammaSum_0, gammaBiword):
        # Update a
        for Len in self.eLengthSet:
            for prev_j in range(Len):
                deltaSum = 0.0
                for j in range(Len):
                    deltaSum += delta[Len][prev_j][j]
                for j in range(Len):
                    self.a[Len][prev_j][j] = delta[Len][prev_j][j] /\
                        (deltaSum + 1e-37)

        # Update pi
        for i in range(maxE):
            self.pi[i] = gammaSum_0[i] * (1.0 / self.lenDataset)

        # Update t
        gammaEWord = defaultdict(float)
        for f, e in gammaBiword:
            gammaEWord[e] += gammaBiword[(f, e)]
        self.t.clear()
        for f, e in gammaBiword:
            self.t[(f, e)] = gammaBiword[(f, e)] / (gammaEWord[e] + 1e-37)
        return

    def endOfBaumWelch(self):
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
        alignerIBM1.initialiseBiwordCount(dataset)
        alignerIBM1.EM(dataset, iterations, 'IBM1')
        self.t = alignerIBM1.t
        self.task.progress("IBM model Trained")
        self.logger.info("IBM model Trained")
        self.baumWelch(dataset, iterations=iterations)
        self.task.progress("finalising")
        self.task = None
        return
