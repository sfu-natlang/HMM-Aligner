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
from models.IBM1WithAlignmentType import AlignmentModel as AlignerIBM1Type
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
except all:
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

    def train(self, dataset, iterations):
        self.task = Task("Aligner", "HMMOI" + str(iterations))
        self.task.progress("Training IBM model 1")
        self.logger.info("Training IBM model 1")
        alignerIBM1 = AlignerIBM1Type()
        alignerIBM1.initialiseAlignTypeDist(dataset, self.loadTypeDist)
        self.typeList, self.typeIndex, self.typeDist =\
            alignerIBM1.typeList, alignerIBM1.typeIndex, alignerIBM1.typeDist
        alignerIBM1.trainStage2(dataset, iterations)
        self.t = alignerIBM1.t
        self.s = alignerIBM1.s
        self.task.progress("IBM model Trained")
        self.logger.info("IBM model Trained")
        self.baumWelch(dataset, iterations=iterations)
        self.task.progress("finalising")
        self.task = None
        return

    def logViterbi(self, f, e):
        e = deepcopy(e)
        fLen, eLen = len(f), len(e)
        for i in range(eLen):
            e.append(("null", "null"))

        score = [[[0.0 for z in range(len(self.typeList))]
                  for y in range(eLen * 2)] for x in range(fLen)]
        prev_j = [[[0 for z in range(len(self.typeList))]
                   for y in range(eLen * 2)] for x in range(fLen)]
        prev_h = [[[0 for z in range(len(self.typeList))]
                   for y in range(eLen * 2)] for x in range(fLen)]

        for i in range(fLen):
            for j in range(eLen * 2):
                tPr = log(self.tProbability(f[i], e[j]))
                for h in range(len(self.typeList)):
                    sPr = log(self.sProbability(f[i], e[j], h))
                    score[i][j][h] = tPr + sPr
                if i == 0:
                    if j < len(self.pi) and self.pi[j] != 0:
                        score[i][j][h] += log(self.pi[j])
                    else:
                        score[i][j][h] = - sys.maxint - 1
                else:
                    # Find the best alignment for f[i-1]
                    maxScore = -sys.maxint - 1
                    bestPrev_j = -sys.maxint - 1
                    bestPrev_h = 0
                    for jPrev in range(eLen * 2):
                        aPrPreLog = self.aProbability(jPrev, j, eLen)
                        if aPrPreLog == 0:
                            continue
                        aPr = log(aPrPreLog)
                        for h in range(len(self.typeList)):
                            temp = score[i - 1][jPrev][h] + aPr
                            if temp > maxScore:
                                maxScore = temp
                                bestPrev_j = jPrev
                                bestPrev_h = h

                    for h in range(len(self.typeList)):
                        score[i][j][h] += maxScore
                        prev_j[i][j][h] = bestPrev_j
                        prev_h[i][j][h] = bestPrev_h

        maxScore = -sys.maxint - 1
        best_j = best_h = 0
        for j in range(eLen * 2):
            for h in range(len(self.typeList)):
                if score[-1][j][h] > maxScore:
                    maxScore = score[-1][j][h]
                    best_j, best_h = j, h

        trace = [best_j, ]
        bestLinkTrace = [best_h, ]
        i = fLen - 1
        j, h = best_j, best_h

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
