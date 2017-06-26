# -*- coding: utf-8 -*-

#
# HMM model implementation(old) of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of HMM word aligner, it requires IBM1Old in order
# to function properly
#
import optparse
import sys
import os
import logging
import time
import numpy as np
from math import log
from collections import defaultdict
from copy import deepcopy

from loggers import logging
from models.IBM1 import AlignmentModel as AlignerIBM1
from models.modelBase import AlignmentModelBase as Base
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
        self.logger = logging.getLogger('HMM')
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.a = None
        self.pi = None
        self.task = None
        self.evaluate = evaluate
        self.modelComponents = ["t", "pi", "a"]
        Base.__init__(self)
        return

    def initialiseModel(self, Len):
        doubleLen = 2 * Len
        # a: transition parameter
        # pi: initial parameter
        tmp = 1.0 / Len
        for z in range(doubleLen + 1):
            for y in range(doubleLen + 1):
                for x in range(Len + 1):
                    self.a[z][y][x] = tmp
        tmp = 1.0 / doubleLen
        for x in range(doubleLen + 1):
            self.pi[x] = tmp
        return

    def forwardBackward(self, f, e, small_t):
        alphaScale = [0.0 for x in range(len(f) + 1)]
        alpha = [[0.0 for x in range(len(f) + 1)] for y in range(len(e) + 1)]
        alphaSum = 0

        for i in range(1, len(e) + 1):
            alpha[i][1] = self.pi[i] * small_t[0][i - 1]
            alphaSum += alpha[i][1]

        alphaScale[1] = 1 / alphaSum
        for i in range(1, len(e) + 1):
            alpha[i][1] *= alphaScale[1]

        for t in range(2, len(f) + 1):
            alphaSum = 0
            for j in range(1, len(e) + 1):
                total = 0
                for i in range(1, len(e) + 1):
                    total += alpha[i][t - 1] * self.a[i][j][len(e)]
                alpha[j][t] = small_t[t - 1][j - 1] * total
                alphaSum += alpha[j][t]

            alphaScale[t] = 1.0 / alphaSum
            for i in range(1, len(e) + 1):
                alpha[i][t] = alphaScale[t] * alpha[i][t]

        beta = [[0.0 for x in range(len(f) + 1)] for y in range(len(e) + 1)]
        for i in range(1, len(e) + 1):
            beta[i][len(f)] = alphaScale[len(f)]
        for t in range(len(f) - 1, 0, -1):
            for i in range(1, len(e) + 1):
                total = 0
                for j in range(1, len(e) + 1):
                    total += (beta[j][t + 1] *
                              self.a[i][j][len(e)] *
                              small_t[t][j - 1])
                beta[i][t] = alphaScale[t] * total
        return alpha, alphaScale, beta

    def maxTargetSentenceLength(self, dataset):
        maxLength = 0
        targetLengthSet = defaultdict(int)
        for (f, e, alignment) in dataset:
            tempLength = len(e)
            if tempLength > maxLength:
                maxLength = tempLength
            targetLengthSet[tempLength] += 1
        return (maxLength, targetLengthSet)

    def mapBitextToInt(self, sd_count):
        index = defaultdict(int)
        biword = defaultdict(tuple)
        i = 0
        for key in sd_count:
            index[key] = i
            biword[i] = key
            i += 1
        return (index, biword)

    def baumWelch(self, dataset, iterations=5):
        if not self.task:
            self.task = Task("Aligner", "HMMBaumWelchOI" + str(iterations))
        N, self.targetLengthSet = self.maxTargetSentenceLength(dataset)

        self.logger.info("N " + str(N))
        indexMap, biword = self.mapBitextToInt(self.t)

        sd_size = len(indexMap)
        totalGammaDeltaOAO_t_i = None
        totalGammaDeltaOAO_t_overall_states_over_dest = None

        twoN = 2 * N

        self.a = [[[0.0 for x in range(N + 1)]
                  for y in range(twoN + 1)]
                  for z in range(twoN + 1)]
        self.pi = [0.0 for x in range(twoN + 1)]

        for iteration in range(iterations):
            self.logger.info("HMMBWTypeI Iteration " + str(iteration))

            logLikelihood = 0

            totalGammaDeltaOAO_t_i = \
                [0.0 for x in range(sd_size)]
            totalGammaDeltaOAO_t_overall_states_over_dest =\
                defaultdict(float)
            totalGamma1OAO = [0.0 for x in range(N + 1)]
            totalC_j_Minus_iOAO = [[[0.0 for x in range(N + 1)]
                                    for y in range(N + 1)]
                                   for z in range(N + 1)]
            totalC_l_Minus_iOAO = [[0.0 for x in range(N + 1)]
                                   for y in range(N + 1)]

            gamma = \
                [[0.0 for x in range(N * 2 + 1)] for y in range(N + 1)]
            small_t = \
                [[0.0 for x in range(N * 2 + 1)] for y in range(N * 2 + 1)]

            start0_time = time.time()

            counter = 0
            for (f, e, alignment) in dataset:
                self.task.progress("BaumWelch iter %d, %d of %d" %
                                   (iteration, counter, len(dataset),))
                counter += 1
                c = defaultdict(float)
                if iteration == 0:
                    self.initialiseModel(len(e))
                for i in range(len(f)):
                    for j in range(len(e)):
                        small_t[i][j] = self.t[(f[i][0], e[j][0])]
                alpha, alphaScaled, beta = self.forwardBackward(f, e, small_t)

                # Setting gamma
                for t in range(1, len(f) + 1):
                    logLikelihood += -1 * log(alphaScaled[t])
                    for i in range(1, len(e) + 1):
                        gamma[i][t] =\
                            (alpha[i][t] * beta[i][t]) / alphaScaled[t]
                        totalGammaDeltaOAO_t_i[
                            indexMap[(f[t - 1][0], e[i - 1][0])]] +=\
                            gamma[i][t]
                logLikelihood += -1 * log(alphaScaled[len(f)])

                for t in range(1, len(f)):
                    for i in range(1, len(e) + 1):
                        for j in range(1, len(e) + 1):
                            c[j - i] += (alpha[i][t] *
                                         self.a[i][j][len(e)] *
                                         small_t[t][j - 1] *
                                         beta[j][t + 1])

                for i in range(1, len(e) + 1):
                    for j in range(1, len(e) + 1):
                        totalC_j_Minus_iOAO[i][j][len(e)] += c[j - i]
                        totalC_l_Minus_iOAO[i][len(e)] += c[j - i]
                    totalGamma1OAO[i] += gamma[i][1]
            # end of loop over dataset

            start_time = time.time()

            self.logger.info("likelihood " + str(logLikelihood))
            N = len(totalGamma1OAO) - 1

            for k in range(sd_size):
                f, e = biword[k]

                totalGammaDeltaOAO_t_overall_states_over_dest[e] +=\
                    totalGammaDeltaOAO_t_i[k]

            end_time = time.time()

            self.logger.info("time spent in the end of E-step: " +
                             str(end_time - start_time))
            self.logger.info("time spent in E-step: " +
                             str(end_time - start0_time))

            # M-Step
            del self.a
            del self.pi
            del self.t
            self.a = [[[0.0 for x in range(N + 1)]
                      for y in range(twoN + 1)]
                      for z in range(twoN + 1)]
            self.pi = [0.0 for x in range(twoN + 1)]
            self.t = defaultdict(float)

            self.logger.info("set " + str(self.targetLengthSet.keys()))
            for I in self.targetLengthSet:
                for i in range(1, I + 1):
                    for j in range(1, I + 1):
                        self.a[i][j][I] = \
                            totalC_j_Minus_iOAO[i][j][I] /\
                            (totalC_l_Minus_iOAO[i][I] + 1e-37)

            for i in range(1, N + 1):
                self.pi[i] = totalGamma1OAO[i] * (1.0 / len(dataset))

            for k in range(sd_size):
                f, e = biword[k]
                self.t[(f, e)] = totalGammaDeltaOAO_t_i[k] / \
                    (totalGammaDeltaOAO_t_overall_states_over_dest[e] + 1e-37)

            end2_time = time.time()
            self.logger.info("time spent in M-step: " +
                             str(end2_time - end_time))
            self.logger.info("iteration " + str(iteration) + " completed")
        return

    def multiplyOneMinusP0H(self):
        for I in self.targetLengthSet:
            for i in range(1, I + 1):
                for j in range(1, I + 1):
                    self.a[i][j][I] *= 1 - self.p0H
        for I in self.targetLengthSet:
            for i in range(1, I + 1):
                for j in range(1, I + 1):
                    self.a[i][i + I][I] = self.p0H
                    self.a[i + I][i + I][I] = self.p0H
                    self.a[i + I][j][I] = self.a[i][j][I]
        return

    def tProbability(self, f, e):
        v = 163303
        if (f[0], e[0]) in self.t:
            return self.t[(f[0], e[0])]
        if e[0] == "null":
            return self.nullEmissionProb
        return 1.0 / v

    def aProbability(self, prev_j, i, targetLength):
        # p(i|i',I) is smoothed to uniform distribution for now -->
        # p(i|i',I) = 1/I
        # we can make it interpolation form like what Och and Ney did
        if targetLength in self.targetLengthSet:
            return self.a[prev_j + 1][i + 1][targetLength] + 1e-64
        return 1.0 / targetLength

    def logViterbi(self, f, e):
        '''
        This function returns alignment of given sentence in two languages
        @param f: source sentence
        @param e: target sentence
        @return: list of alignment
        '''
        e = deepcopy(e)
        eLen = len(e)
        for i in range(eLen):
            e.append(("null", "null"))
        score = np.zeros((len(f), len(e)))
        prev_j = np.zeros((len(f), len(e)))

        for i in range(len(f)):
            for j in range(len(e)):
                score[i][j] = log(self.tProbability(f[i], e[j]))
                if i == 0:
                    if j + 1 < len(self.pi) and self.pi[j + 1] != 0:
                        score[i][j] += log(self.pi[j + 1])
                    else:
                        score[i][j] = - sys.maxint - 1
                else:
                    # Find the best alignment for f[i-1]
                    maxScore = -sys.maxint - 1
                    bestPrev_j = -sys.maxint - 1
                    for jPrev in range(len(e)):
                        temp = score[i - 1][jPrev] +\
                            log(self.aProbability(jPrev, j, eLen))
                        if temp > maxScore:
                            maxScore = temp
                            bestPrev_j = jPrev

                    score[i][j] += maxScore
                    prev_j[i][j] = bestPrev_j

        maxScore = -sys.maxint - 1
        best_j = 0
        for j in range(len(e)):
            if score[len(f) - 1][j] > maxScore:
                maxScore = score[len(f) - 1][j]
                best_j = j

        trace = [best_j + 1, ]
        i = len(f) - 1
        j = best_j

        while (i > 0):
            j = int(prev_j[i][j])
            trace = [j + 1] + trace
            i = i - 1
        return trace

    def decode(self, dataset):
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(dataset)))
        result = []

        for (f, e, alignment) in dataset:
            sentenceAlignment = []
            N = len(e)
            bestAlignment = self.logViterbi(f, e)

            for i in range(len(bestAlignment)):
                if bestAlignment[i] <= len(e):
                    sentenceAlignment.append((i + 1, bestAlignment[i]))

            result.append(sentenceAlignment)
        self.logger.info("Decoding Completed")
        return result

    def train(self, dataset, iterations):
        self.task = Task("Aligner", "HMMOI" + str(iterations))
        self.task.progress("Training IBM model 1")
        self.logger.info("Training IBM model 1")
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.train(dataset, iterations)
        self.t = alignerIBM1.t
        self.task.progress("IBM model Trained")
        self.logger.info("IBM model Trained")
        self.baumWelch(dataset, iterations=iterations)
        self.task.progress("finalising")
        self.multiplyOneMinusP0H()
        self.task = None
        return
