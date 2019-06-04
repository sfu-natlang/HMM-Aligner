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
from models.IBM1 import AlignmentModel as AlignerIBM1
from models.HMM import AlignmentModel as HMM
from evaluators.evaluator import evaluate
__version__ = "0.5a"


class AlignmentModel(HMM):
    def __init__(self):
        HMM.__init__(self)
        self.modelName = "HMMWithAlignmentType"
        self.version = "0.4b"

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

        self.loadTypeDist = {"SEM": .401, "FUN": .264, "PDE": .004,
                             "CDE": .004, "MDE": .012, "GIS": .205,
                             "GIF": .031, "COI": .008, "TIN": .003,
                             "NTR": .086, "MTA": .002}

        self.modelComponents = ["t", "pi", "a", "eLengthSet", "s", "sTag",
                                "typeList", "typeIndex", "typeDist",
                                "fLex", "eLex", "fIndex", "eIndex",
                                "lambd", "lambda1", "lambda2", "lambda3"]
        return

    def _beginningOfIteration(self, dataset, maxE, index):
        HMM._beginningOfIteration(self, dataset, maxE, index)
        self.c_feh = [defaultdict(lambda: np.zeros(len(self.typeIndex)))
                      for i in range(len(self.fLex[index]))]
        return

    def EStepGamma(self, f, e, gamma, index):
        HMM.EStepGamma(self, f, e, gamma, index)
        score = self.sProbability(f, e, index) * gamma[:, :, None]
        for i in range(len(f)):
            for j in range(len(e)):
                self.c_feh[f[i][index]][e[j][index]] += score[i][j]
        return

    def MStepGamma(self, maxE, index):
        HMM.MStepGamma(self, maxE, index)
        # Update s
        if self.index == 0:
            self.s = self.c_feh
        else:
            self.sTag = self.c_feh
        for i in range(len(self.c_feh)):
            for j in self.c_feh[i]:
                self.c_feh[i][j] /= self.gammaBiword[i][j]
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
        alignerIBM1.EM(dataset, iterations, index)
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
        return

    def train(self, dataset, iterations=5):
        dataset = self.initialiseLexikon(dataset)
        self.logger.info("Loading alignment type distribution")
        self.initialiseAlignTypeDist(dataset, self.loadTypeDist)
        self.logger.info("Alignment type distribution loaded")

        self.logger.info("Stage 1 Training With POS Tags")
        self.trainWithIndex(dataset, iterations, 1)

        self.logger.info("Stage 2 Training With FORM")
        self.trainWithIndex(dataset, iterations, 0)

        self.logger.info("Training Complete")
        return

    def logViterbi(self, f, e):
        e = deepcopy(e)
        with np.errstate(invalid='ignore', divide='ignore'):
            a = np.log(self.aProbability(f, e))
        fLen, eLen = len(f), len(e)
        for i in range(eLen):
            e.append((424242424243, 424242424243))
        score = np.zeros((fLen, eLen * 2))
        prev_j = np.zeros((fLen, eLen * 2))
        s = self.sProbability(f, e)
        types = np.argmax(s, axis=2)

        with np.errstate(invalid='ignore', divide='ignore'):
            score = np.log(self.tProbability(f, e)) + np.log(np.max(s, axis=2))
        for i in range(fLen):
            if i == 0:
                with np.errstate(invalid='ignore', divide='ignore'):
                    if 2 * eLen <= self.pi.shape[0]:
                        score[i] += np.log(self.pi[:eLen * 2])
                    else:
                        score[i][:self.pi.shape[0]] += np.log(self.pi)
                        score[i][self.pi.shape[0]:].fill(-sys.maxsize)
            else:
                tmp = (a[i].T + score[i - 1]).T
                bestPrev_j = np.argmax(tmp, axis=0)
                prev_j[i] = bestPrev_j
                score[i] += np.max(tmp, axis=0)

        maxScore = -sys.maxsize - 1
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
        score[:, eLen] = np.max(score[:, eLen:], axis=1)
        return trace, score[:, :eLen + 1]
