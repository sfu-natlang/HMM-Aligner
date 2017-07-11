# -*- coding: utf-8 -*-

#
# HMM model with alignment type implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of HMM word aligner, it requires IBM1 in order to
# function properly
#
import sys
import numpy as np
from math import log
from collections import defaultdict
from copy import deepcopy

from loggers import logging
from models.cIBM1 import AlignmentModel as AlignerIBM1
from models.cModelBase import Task
from models.cHMMBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(Base):
    def __init__(self):
        self.modelName = "HMMWithAlignmentType"
        self.version = "0.3b"
        self.logger = logging.getLogger('HMM')
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.task = None
        self.evaluate = evaluate
        self.fe = ()

        self.s = []
        self.sTag = []
        self.index = 0
        self.typeList = []
        self.typeIndex = {}
        self.typeDist = []
        self.lambd = 1 - 1e-20
        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15
        self.fLex = self.eLex = self.fIndex = self.eIndex = None

        self.loadTypeDist = {"SEM": .401, "FUN": .264, "PDE": .004,
                             "CDE": .004, "MDE": .012, "GIS": .205,
                             "GIF": .031, "COI": .008, "TIN": .003,
                             "NTR": .086, "MTA": .002}

        self.modelComponents = ["t", "pi", "a", "eLengthSet", "s", "sTag",
                                "typeList", "typeIndex", "typeDist",
                                "fLex", "eLex", "fIndex", "eIndex",
                                "lambd", "lambda1", "lambda2", "lambda3"]
        Base.__init__(self)
        return

    def _beginningOfIteration(self, dataset, maxE, index):
        self.lenDataset = len(dataset)
        self.gammaEWord = [0.0 for i in range(len(self.eLex[index]))]
        self.gammaBiword = [defaultdict(float)
                            for i in range(len(self.fLex[index]))]
        self.gammaSum_0 = np.zeros(maxE)
        self.c_feh = [defaultdict(lambda: np.zeros(len(self.typeIndex)))
                      for i in range(len(self.fLex[index]))]
        return

    def _updateGamma(self, f, e, alpha, beta, alphaScale, index):
        fWords = [f[i][index] for i in range(len(f))]
        eWords = [e[j][index] for j in range(len(e))]
        gamma = ((alpha * beta).T / alphaScale).T
        score = self.sProbability(f, e, index) * gamma[:, :, None]
        for i in range(len(f)):
            for j in range(len(e)):
                self.gammaBiword[fWords[i]][eWords[j]] += gamma[i][j]
                self.gammaEWord[eWords[j]] += gamma[i][j]
                self.c_feh[fWords[i]][eWords[j]] += score[i][j]
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

        # Update s
        if self.index == 0:
            del self.s
            self.s = self.c_feh
        else:
            del self.sTag
            self.sTag = self.c_feh
        for i in range(len(self.c_feh)):
            for j in self.c_feh[i]:
                self.c_feh[i][j] /= self.gammaBiword[i][j]
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

    def sProbability(self, f, e, index=0):
        sTag = np.tile((1 - self.lambd) * self.typeDist, (len(f), len(e), 1))

        for j in range(len(e)):
            for i in range(len(f)):
                if f[i][1] < len(self.sTag) and e[j][1] in self.sTag[f[i][1]]:
                    sTag[i][j] += self.lambd * self.sTag[f[i][1]][e[j][1]]
        if index == 1:
            return sTag

        s = np.tile((1 - self.lambd) * self.typeDist, (len(f), len(e), 1))
        for j in range(len(e)):
            for i in range(len(f)):
                if f[i][0] < len(self.s) and e[j][0] in self.s[f[i][0]]:
                    s[i][j] += self.lambd * self.s[f[i][0]][e[j][0]]

        return (self.lambda1 * s +
                self.lambda2 * sTag +
                np.tile(self.lambda3 * self.typeDist, (len(f), len(e), 1)))

    def trainWithIndex(self, dataset, iterations, index):
        self.index = index
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.sharedLexikon(self)
        alignerIBM1.initialiseBiwordCount(dataset, index)
        alignerIBM1.EM(dataset, iterations, 'IBM1', index)
        self.task.progress("IBM model Trained")
        self.logger.info("IBM model Trained")

        self.logger.info("Initialising HMM")
        self.initialiseBiwordCount(dataset, index)
        if self.index == 1:
            self.sTag = self.calculateS(dataset, index)
        else:
            self.s = self.calculateS(dataset, index)
        self.t = alignerIBM1.t
        self.logger.info("HMM Initialised, start training")
        self.baumWelch(dataset, iterations=iterations, index=index)
        self.task.progress("HMM finalising")
        return

    def train(self, dataset, iterations=5):
        dataset = self.initialiseLexikon(dataset)
        self.task = Task("Aligner", "HMMOI" + str(iterations))
        self.logger.info("Loading alignment type distribution")
        self.initialiseAlignTypeDist(dataset, self.loadTypeDist)
        self.logger.info("Alignment type distribution loaded")

        self.task.progress("Stage 1 Training With POS Tags")
        self.logger.info("Stage 1 Training With POS Tags")
        self.trainWithIndex(dataset, iterations, 1)

        self.task.progress("Stage 2 Training With FORM")
        self.logger.info("Stage 2 Training With FORM")
        self.trainWithIndex(dataset, iterations, 0)

        self.logger.info("Training Complete")
        self.task = None
        return

    def logViterbi(self, f, e):
        e = deepcopy(e)
        fLen, eLen = len(f), len(e)
        for i in range(eLen):
            e.append((424242424243, 424242424243))
        score = np.zeros((fLen, eLen * 2))
        prev_j = np.zeros((fLen, eLen * 2))
        s = self.sProbability(f, e)
        types = np.argmax(s, axis=2)

        with np.errstate(invalid='ignore', divide='ignore'):
            score = np.log(self.tProbability(f, e)) + np.log(np.max(s, axis=2))
            a = np.log(self.aProbability(eLen))
        for i in range(fLen):
            if i == 0:
                with np.errstate(invalid='ignore', divide='ignore'):
                    if 2 * eLen <= self.pi.shape[0]:
                        score[i] += np.log(self.pi[:eLen * 2])
                    else:
                        score[i][:self.pi.shape[0]] += np.log(self.pi)
                        score[i][self.pi.shape[0]:].fill(-sys.maxint)
            else:
                tmp = (a.T + score[i - 1]).T
                bestPrev_j = np.argmax(tmp, axis=0)
                prev_j[i] = bestPrev_j
                score[i] += np.max(tmp, axis=0)

        maxScore = -sys.maxint - 1
        best_j = 0
        for j in range(eLen * 2):
            if score[fLen - 1][j] > maxScore:
                maxScore = score[fLen - 1][j]
                best_j = j

        i = fLen - 1
        j = best_j
        trace = [(j + 1, int(types[i][j]))]

        while (i > 0):
            j = int(prev_j[i][j])
            i = i - 1
            trace = [(j + 1, int(types[i][j]))] + trace
        return trace
