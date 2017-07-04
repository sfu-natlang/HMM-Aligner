# -*- coding: utf-8 -*-

#
# HMM model base of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the base model for HMM
#
import sys
import time
import numpy as np
from math import log
from collections import defaultdict
from copy import deepcopy

from loggers import logging
from models.modelBase import Task
from models.modelBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModelBase(Base):
    def __init__(self):
        if "nullEmissionProb" not in vars(self):
            self.nullEmissionProb = 0.000005
        if "task" not in vars(self):
            self.task = None

        if "t" not in vars(self):
            self.t = defaultdict(float)
        if "eLengthSet" not in vars(self):
            self.eLengthSet = defaultdict(int)
        if "a" not in vars(self):
            self.a = [[[]]]
        if "pi" not in vars(self):
            self.pi = []

        if "logger" not in vars(self):
            self.logger = logging.getLogger('HMMBASE')
        if "modelComponents" not in vars(self):
            self.modelComponents = ["t", "pi", "a", "eLengthSet"]
        Base.__init__(self)
        return

    def initialiseParameter(self, Len):
        doubleLen = 2 * Len
        tmp = 1.0 / Len
        for z in range(Len):
            for y in range(Len):
                for x in range(Len + 1):
                    self.a[x][z][y] = tmp
        tmp = 1.0 / doubleLen
        for x in range(Len):
            self.pi[x] = tmp
        return

    def forwardBackward(self, f, e, tSmall, a):
        alpha = [[0.0 for x in range(len(e))] for y in range(len(f))]
        alphaScale = [0.0 for x in range(len(f))]
        alphaSum = 0

        for j in range(len(e)):
            alpha[0][j] = self.pi[j] * tSmall[0][j]
            alphaSum += alpha[0][j]

        alphaScale[0] = 1 / alphaSum
        for j in range(len(e)):
            alpha[0][j] *= alphaScale[0]

        for i in range(1, len(f)):
            alphaSum = 0
            for j in range(len(e)):
                total = 0
                for prev_j in range(len(e)):
                    total += alpha[i - 1][prev_j] * a[prev_j][j]
                alpha[i][j] = tSmall[i][j] * total
                alphaSum += alpha[i][j]

            alphaScale[i] = 1.0 / alphaSum
            for j in range(len(e)):
                alpha[i][j] = alphaScale[i] * alpha[i][j]

        beta = [[0.0 for x in range(len(e))] for y in range(len(f))]
        for j in range(len(e)):
            beta[len(f) - 1][j] = alphaScale[len(f) - 1]

        for i in range(len(f) - 2, -1, -1):
            for j in range(len(e)):
                total = 0
                for next_j in range(len(e)):
                    total += (beta[i + 1][next_j] * a[j][next_j] *
                              tSmall[i + 1][next_j])
                beta[i][j] = alphaScale[i] * total
        return alpha, alphaScale, beta

    def maxTargetSentenceLength(self, dataset):
        maxLength = 0
        eLengthSet = defaultdict(int)
        for (f, e, alignment) in dataset:
            tempLength = len(e)
            if tempLength > maxLength:
                maxLength = tempLength
            eLengthSet[tempLength] += 1
        return (maxLength, eLengthSet)

    def baumWelch(self, dataset, iterations=5, index=0):
        if not self.task:
            self.task = Task("Aligner", "HMMBaumWelchOI" + str(iterations))
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(dataset)))
        startTime = time.time()

        maxE, self.eLengthSet = self.maxTargetSentenceLength(dataset)
        self.logger.info("Maximum Target sentence length: " + str(maxE))

        self.a = [[[0.0 for x in range(maxE * 2)] for y in range(maxE * 2)]
                  for z in range(maxE + 1)]
        self.pi = [0.0 for x in range(maxE * 2)]

        for iteration in range(iterations):
            self.logger.info("BaumWelch Iteration " + str(iteration))

            logLikelihood = 0

            gamma = [[0.0 for x in range(maxE)] for y in range(maxE * 2)]
            gammaBiword = defaultdict(float)
            gammaSum_0 = [0.0 for x in range(maxE)]
            delta = [[[0.0 for x in range(maxE)] for y in range(maxE)]
                     for z in range(maxE + 1)]

            self._beginningOfIteration(dataset)

            counter = 0
            for (f, e, alignment) in dataset:
                self.task.progress("BaumWelch iter %d, %d of %d" %
                                   (iteration, counter, len(dataset),))
                counter += 1
                if iteration == 0:
                    self.initialiseParameter(len(e))

                fLen, eLen = len(f), len(e)
                a = self.a[eLen]
                tSmall = [[self.t[(f[i][index], e[j][index])]
                           for j in range(eLen)]
                          for i in range(fLen)]

                alpha, alphaScale, beta = self.forwardBackward(f, e, tSmall, a)

                # Update logLikelihood
                for i in range(fLen):
                    logLikelihood -= log(alphaScale[i])

                # Setting gamma
                self._updateGamma(f, e, gamma, alpha, beta, alphaScale)

                for i in range(fLen):
                    for j in range(eLen):
                        gammaBiword[(f[i][index], e[j][index])] += gamma[i][j]
                for j in range(eLen):
                    gammaSum_0[j] += gamma[0][j]

                # Update delta
                c = [0.0 for i in range(eLen * 2)]
                for i in range(1, fLen):
                    for prev_j in range(eLen):
                        for j in range(eLen):
                            c[eLen - 1 + j - prev_j] += (alpha[i - 1][prev_j] *
                                                         beta[i][j] *
                                                         a[prev_j][j] *
                                                         tSmall[i][j])

                for prev_j in range(eLen):
                    for j in range(eLen):
                        delta[eLen][prev_j][j] += c[eLen - 1 + j - prev_j]
            # end of loop over dataset

            self.logger.info("likelihood " + str(logLikelihood))
            # M-Step
            self._updateEndOfIteration(maxE, delta, gammaSum_0, gammaBiword)

        self.endOfBaumWelch()
        endTime = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (endTime - startTime,))
        return

    def _beginningOfIteration(self, dataset):
        # self.lenDataset = len(dataset)
        # return
        raise NotImplementedError

    def _updateGamma(self, f, e, gamma, alpha, beta, alphaScale):
        # for i in range(len(f)):
        #     for j in range(len(e)):
        #         gamma[i][j] = alpha[i][j] * beta[i][j] / alphaScale[i]
        raise NotImplementedError

    def _updateEndOfIteration(self, maxE, delta, gammaSum_0, gammaBiword):
        # self.t.clear()
        # for Len in self.eLengthSet:
        #     for prev_j in range(Len):
        #         deltaSum = 0.0
        #         for j in range(Len):
        #             deltaSum += delta[Len][prev_j][j]
        #         for j in range(Len):
        #             self.a[Len][prev_j][j] = delta[Len][prev_j][j] /\
        #                 (deltaSum + 1e-37)

        # for i in range(maxE):
        #     self.pi[i] = gammaSum_0[i] * (1.0 / self.lenDataset)

        # gammaEWord = defaultdict(float)
        # for f, e in gammaBiword:
        #     gammaEWord[e] += gammaBiword[(f, e)]
        # for f, e in gammaBiword:
        #     self.t[(f, e)] = gammaBiword[(f, e)] / (gammaEWord[e] + 1e-37)
        # return
        raise NotImplementedError

    def endOfBaumWelch(self):
        # Apply final smoothing here
        raise NotImplementedError

    def tProbability(self, f, e, index=0):
        v = 163303
        if (f[index], e[index]) in self.t:
            return self.t[(f[index], e[index])]
        if e[index] == "null":
            return self.nullEmissionProb
        return 1.0 / v

    def aProbability(self, prev_j, j, targetLength):
        if targetLength in self.eLengthSet:
            return self.a[targetLength][prev_j][j]
        return 1.0 / targetLength

    def logViterbi(self, f, e):
        e = deepcopy(e)
        fLen, eLen = len(f), len(e)
        for i in range(eLen):
            e.append(("null", "null"))
        score = np.zeros((fLen, eLen * 2))
        prev_j = np.zeros((fLen, eLen * 2))

        for i in range(fLen):
            for j in range(eLen * 2):
                score[i][j] = log(self.tProbability(f[i], e[j]))
                if i == 0:
                    if j < len(self.pi) and self.pi[j] != 0:
                        score[i][j] += log(self.pi[j])
                    else:
                        score[i][j] = - sys.maxint - 1
                else:
                    # Find the best alignment for f[i-1]
                    maxScore = -sys.maxint - 1
                    bestPrev_j = -sys.maxint - 1
                    for jPrev in range(eLen * 2):
                        aPr = self.aProbability(jPrev, j, eLen)
                        if aPr == 0:
                            continue
                        temp = score[i - 1][jPrev] + log(aPr)
                        if temp > maxScore:
                            maxScore = temp
                            bestPrev_j = jPrev

                    score[i][j] += maxScore
                    prev_j[i][j] = bestPrev_j

        maxScore = -sys.maxint - 1
        best_j = 0
        for j in range(eLen * 2):
            if score[fLen - 1][j] > maxScore:
                maxScore = score[fLen - 1][j]
                best_j = j

        trace = [(best_j + 1, )]
        i = fLen - 1
        j = best_j

        while (i > 0):
            j = int(prev_j[i][j])
            trace = [(j + 1, )] + trace
            i = i - 1
        return trace

    def decodeSentence(self, sentence):
        f, e, alignment = sentence
        sentenceAlignment = []
        bestAlign = self.logViterbi(f, e)

        for i in range(len(bestAlign)):

            if bestAlign[i][0] <= len(e):
                if len(bestAlign[i]) > 1 and "typeList" in vars(self):
                    sentenceAlignment.append(
                        (i + 1, bestAlign[i][0],
                         self.typeList[bestAlign[i][1]]))
                else:
                    sentenceAlignment.append((i + 1, bestAlign[i][0]))
        return sentenceAlignment
