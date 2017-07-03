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
__version__ = "0.5a"


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
            self.modelComponents = ["t", "pi", "a", "eLengthSet",
                                    "fLex", "eLex", "fIndex", "eIndex"]
        Base.__init__(self)
        return

    def initialiseParameter(self, Len):
        self.a[:Len + 1, :Len, :Len].fill(1.0 / Len)
        self.pi[:Len].fill(1.0 / 2 / Len)
        return

    def forwardBackward(self, f, e, tSmall, a):
        alpha = np.zeros((len(f), len(e)))
        beta = np.zeros((len(f), len(e)))
        alphaScale = np.zeros(len(f))

        alpha[0] = tSmall[0] * self.pi[:len(e)]
        alphaScale[0] = 1 / np.sum(alpha[0])
        alpha[0] *= alphaScale[0]
        for i in range(1, len(f)):
            alpha[i] = tSmall[i] * np.matmul(alpha[i - 1], a)
            alphaScale[i] = 1 / np.sum(alpha[i])
            alpha[i] *= alphaScale[i]

        beta[len(f) - 1].fill(alphaScale[len(f) - 1])
        for i in range(len(f) - 2, -1, -1):
            beta[i] =\
                np.matmul(beta[i + 1] * tSmall[i + 1], a.T) * alphaScale[i]
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

        self.a = np.zeros((maxE + 1, maxE * 2, maxE * 2))
        self.pi = np.zeros(maxE * 2)

        for iteration in range(iterations):
            self.logger.info("BaumWelch Iteration " + str(iteration))

            logLikelihood = 0

            delta = np.zeros((maxE + 1, maxE, maxE))

            self._beginningOfIteration(dataset, maxE)

            counter = 0
            for (f, e, alignment) in dataset:
                self.task.progress("BaumWelch iter %d, %d of %d" %
                                   (iteration, counter, len(dataset),))
                counter += 1
                if iteration == 0:
                    self.initialiseParameter(len(e))

                fLen, eLen = len(f), len(e)
                fWords = [f[i][index] for i in range(fLen)]
                eWords = [e[j][index] for j in range(eLen)]
                a = self.a[eLen][:len(e), :len(e)]
                tSmall = self.t[fWords][:, eWords]

                alpha, alphaScale, beta = self.forwardBackward(f, e, tSmall, a)

                # Update logLikelihood
                logLikelihood -= np.sum(np.log(alphaScale))

                # Setting gamma
                gamma = self.gamma(f, e, alpha, beta, alphaScale, index)

                # Update delta, the code below is the slow version. It is given
                # here as the matrix version might be difficult to understand
                # at first sight
                # c = [0.0 for i in range(eLen * 2)]
                # for i in range(1, fLen):
                #     for prev_j in range(eLen):
                #         for j in range(eLen):
                #             c[eLen - 1 + j - prev_j] += (
                #                 alpha[i - 1][prev_j] *
                #                 beta[i][j] *
                #                 a[prev_j][j] *
                #                 tSmall[i][j])
                # for prev_j in range(eLen):
                #     for j in range(eLen):
                #         delta[eLen][prev_j][j] += c[eLen - 1 + j - prev_j]
                Xceta = np.matmul(alpha[:fLen - 1].T, (beta * tSmall)[1:]) * a
                c = np.zeros(eLen * 2)
                for j in range(eLen):
                    c[eLen - 1 - j:2 * eLen - 1 - j] += Xceta[j]
                for j in range(eLen):
                    delta[eLen][j][:eLen] +=\
                        c[eLen - 1 - j:2 * eLen - 1 - j]
            # end of loop over dataset

            self.logger.info("likelihood " + str(logLikelihood))
            # M-Step
            self._updateEndOfIteration(maxE, delta)

        self.endOfBaumWelch()
        endTime = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (endTime - startTime,))
        return

    def _beginningOfIteration(self, dataset, maxE):
        raise NotImplementedError

    def gamma(self, f, e, alpha, beta, alphaScale):
        raise NotImplementedError

    def _updateEndOfIteration(self, maxE, delta):
        raise NotImplementedError

    def endOfBaumWelch(self):
        raise NotImplementedError

    def tProbability(self, f, e, index=0):
        if f[index] < self.t.shape[0] and e[index] < self.t.shape[1]:
            tmp = self.t[f[index]][e[index]]
            if tmp == 0:
                return 0.000006123586217
            else:
                return tmp
        if e[index] == 424242424243:
            return self.nullEmissionProb
        return 0.000006123586217

    def aProbability(self, targetLength):
        if targetLength in self.eLengthSet:
            return self.a[targetLength][:targetLength * 2, :targetLength * 2]
        return np.full((targetLength * 2, targetLength * 2), 1. / targetLength)

    def logViterbi(self, f, e):
        e = deepcopy(e)
        fLen, eLen = len(f), len(e)
        for i in range(eLen):
            e.append((424242424243, 424242424243))
        score = np.zeros((fLen, eLen * 2))
        prev_j = np.zeros((fLen, eLen * 2))

        a = self.aProbability(eLen)
        for i in range(fLen):
            for j in range(eLen * 2):
                score[i][j] = log(self.tProbability(f[i], e[j]))
                if i == 0:
                    if j < len(self.pi) and self.pi[j] != 0:
                        score[i][j] += log(self.pi[j])
                    else:
                        score[i][j] = - sys.maxint - 1
                else:
                    aPrs = a[:, j]
                    with np.errstate(invalid='ignore', divide='ignore'):
                        tmp = score[i - 1] + np.log(aPrs)
                    bestPrev_j = np.argmax(tmp)
                    prev_j[i][j] = bestPrev_j
                    score[i][j] += tmp[bestPrev_j]

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
