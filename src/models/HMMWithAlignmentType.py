# -*- coding: utf-8 -*-

#
# HMM model implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of HMM word aligner, it requires IBM1 in order to
# function properly
#
import sys
from collections import defaultdict
from copy import deepcopy
from math import log
from loggers import logging
from models.IBM1 import AlignmentModel as AlignerIBM1
from models.HMMBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


# This is a private module for transmitting test results. Please ignore.
class DummyTask():
    def __init__(self, taskName="Untitled", serial="XXXX"):
        return

    def progress(self, msg):
        return


try:
    from progress import Task
except ImportError:
    Task = DummyTask


class AlignmentModel(Base):
    def __init__(self):
        self.modelName = "HMMWithAlignmentType"
        self.version = "0.1b"
        self.logger = logging.getLogger('HMM')
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.task = None
        self.evaluate = evaluate
        self.fe = ()

        self.s = defaultdict(list)
        self.sTag = defaultdict(list)
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
                                "lambd", "lambda1", "lambda2", "lambda3"]
        Base.__init__(self)
        return

    def _beginningOfIteration(self, dataset):
        self.lenDataset = len(dataset)
        self.c_feh = defaultdict(
            lambda: [0.0 for h in range(len(self.typeList))])
        return

    def _updateGamma(self, f, e, gamma, alpha, beta, alphaScale):
        for i in range(len(f)):
            for j in range(len(e)):
                tmpGamma = alpha[i][j] * beta[i][j] / alphaScale[i]
                gamma[i][j] = tmpGamma
                c_feh = self.c_feh[(f[i][self.index], e[j][self.index])]
                for h in range(len(self.typeList)):
                    c_feh[h] += tmpGamma * self.sProbability(f[i], e[j], h)

    def _updateEndOfIteration(self, maxE, delta, gammaSum_0, gammaBiword):
        # Update a
        for targetLen in self.eLengthSet:
            a = self.a[targetLen]
            for prev_j in range(len(a)):
                for j in range(len(a[prev_j])):
                    a[prev_j][j] = 0.0
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

        s = self.s if self.index == 0 else self.sTag
        for (f, e) in self.c_feh:
            c_feh = self.c_feh[(f, e)]
            sTmp = s[(f, e)]
            gammaTmp = gammaBiword[(f, e)]
            for h in range(len(self.typeList)):
                sTmp[h] = c_feh[h] / gammaTmp
        self.fe = ()
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

    def sProbability(self, f, e, h):
        fWord, fTag = f
        eWord, eTag = e
        if self.fe != (f, e):
            self.fe, sKey, sTagKey = (f, e), (f[0], e[0]), (f[1], e[1])
            self.sTmp = self.s[sKey] if sKey in self.s else None
            self.sTagTmp = self.sTag[sTagKey] if sTagKey in self.sTag else None
        sTmp = self.sTmp[h] if self.sTmp else 0
        sTagTmp = self.sTagTmp[h] if self.sTagTmp else 0
        if self.index == 0:
            p1 = (1 - self.lambd) * self.typeDist[h] + self.lambd * sTmp
            p2 = (1 - self.lambd) * self.typeDist[h] + self.lambd * sTagTmp
            p3 = self.typeDist[h]
            return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3 * p3
        else:
            return (1 - self.lambd) * self.typeDist[h] + self.lambd * sTagTmp

    def trainWithIndex(self, dataset, iterations, index):
        self.index = index
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.initialiseBiwordCount(dataset, index)
        alignerIBM1.EM(dataset, iterations, 'IBM1', index)
        self.task.progress("IBM model Trained")
        self.logger.info("IBM model Trained")

        self.logger.info("Initialising HMM")
        self.initialiseBiwordCount(dataset, index)
        if self.index == 1:
            self.sTag = self.calculateS(dataset, self.fe_count, index)
        else:
            self.s = self.calculateS(dataset, self.fe_count, index)
        self.t = alignerIBM1.t
        self.logger.info("HMM Initialised, start training")
        self.baumWelch(dataset, iterations=iterations, index=index)
        self.task.progress("HMM finalising")
        return

    def train(self, dataset, iterations=5):
        self.task = Task("Aligner", "HMMOI" + str(iterations))
        self.logger.info("Loading alignment type distribution")
        self.initialiseAlignTypeDist(dataset, self.loadTypeDist)
        self.logger.info("Alignment type distribution loaded")

        self.task.progress("Stage 1 Training With POS Tags")
        self.logger.info("Stage 1 Training With POS Tags")
        self.trainWithIndex(dataset, iterations, 1)

        self.task.progress("Stage 1 Training With FORM")
        self.logger.info("Stage 1 Training With FORM")
        self.trainWithIndex(dataset, iterations, 0)

        self.logger.info("Training Complete")
        self.task = None
        return

    def logViterbi(self, f, e):
        eLen = len(e)
        e = deepcopy(e)
        for i in range(eLen):
            e.append(("null", "null"))

        score = [[[0.0 for z in range(len(self.typeList))]
                  for x in range(len(e))]
                 for y in range(len(f))]
        prev_j = [[[0 for z in range(len(self.typeList))]
                   for x in range(len(e))]
                  for y in range(len(f))]
        prev_h = [[[0 for z in range(len(self.typeList))]
                   for x in range(len(e))]
                  for y in range(len(f))]

        for j in range(len(e)):
            tPr = log(self.tProbability(f[0], e[j]))
            for h in range(len(self.typeList)):
                score[0][j][h] = log(self.sProbability(f[0], e[j], h)) + tPr
                if j + 1 < len(self.pi) and self.pi[j + 1] != 0:
                    score[0][j][h] += log(self.pi[j + 1])
                else:
                    score[0][j][h] = - sys.maxint - 1

        for i in range(1, len(f)):
            for j in range(len(e)):
                maxScore = -sys.maxint - 1
                jPrevBest = -sys.maxint - 1
                hPrevBest = 0
                tPr = log(self.tProbability(f[i], e[j]))
                for jPrev in range(len(e)):
                    aPrPreLog = self.aProbability(jPrev, j, eLen)
                    if aPrPreLog == 0:
                        continue
                    aPr = log(aPrPreLog)
                    for h in range(len(self.typeList)):
                        temp = score[i - 1][jPrev][h] + aPr + tPr
                        if temp > maxScore:
                            maxScore = temp
                            jPrevBest = jPrev
                            hPrevBest = h

                for h in range(len(self.typeList)):
                    s = self.sProbability(f[i], e[j], h)
                    if s != 0:
                        temp_s = log(s)
                        score[i][j][h] = maxScore + temp_s
                        prev_j[i][j][h] = jPrevBest
                        prev_h[i][j][h] = hPrevBest

        maxScore = -sys.maxint - 1
        best_j = best_h = 0
        for j in range(len(e)):
            for h in range(len(self.typeList)):
                if score[len(f) - 1][j][h] > maxScore:
                    maxScore = score[len(f) - 1][j][h]
                    best_j, best_h = j, h

        trace = [best_j + 1, ]
        bestLinkTrace = [best_h, ]

        j, h = best_j, best_h
        i = len(f) - 1

        while (i > 0):
            j, h = prev_j[i][j][h], prev_h[i][j][h]
            trace = [j + 1] + trace
            bestLinkTrace = [h] + bestLinkTrace
            i = i - 1
        return trace, bestLinkTrace

    def decode(self, dataset):
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(dataset)))
        result = []

        for (f, e, alignment) in dataset:
            sentenceAlignment = []
            bestAlign, bestAlignType = self.logViterbi(f, e)

            for i in range(len(bestAlign)):
                if bestAlign[i] <= len(e):
                    sentenceAlignment.append(
                        (i + 1, bestAlign[i], self.typeList[bestAlignType[i]]))

            result.append(sentenceAlignment)
        self.logger.info("Decoding Completed")
        return result
