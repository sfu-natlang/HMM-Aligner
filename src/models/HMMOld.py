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
from math import log
from collections import defaultdict
from models.IBM1Old import AlignmentModel as AlignerIBM1
from loggers import logging
from models.modelBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
logger = logging.getLogger('HMM')
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

    def initWithIBM(self, modelIBM1, bitext):
        self.f_count = modelIBM1.f_count
        self.fe_count = modelIBM1.fe_count
        self.t = modelIBM1.t
        self.bitext = bitext
        logger.info("IBM Model 1 Loaded")
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

    def forwardWithTScaled(self, f, e, small_t):
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

        return (alpha, alphaScale)

    def backwardWithTScaled(self, f, e, alphaScale, small_t):
        betaHat = [[0.0 for x in range(len(f) + 1)] for y in range(len(e) + 1)]
        for i in range(1, len(e) + 1):
            betaHat[i][len(f)] = alphaScale[len(f)]
        for t in range(len(f) - 1, 0, -1):
            for i in range(1, len(e) + 1):
                total = 0
                for j in range(1, len(e) + 1):
                    total += (betaHat[j][t + 1] *
                              self.a[i][j][len(e)] *
                              small_t[t][j - 1])
                betaHat[i][t] = alphaScale[t] * total
        return betaHat

    def maxTargetSentenceLength(self, bitext):
        maxLength = 0
        targetLengthSet = defaultdict(int)
        for (f, e) in bitext:
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

    def baumWelch(self, iterations=5):
        if not self.task:
            self.task = Task("Aligner", "HMMBaumWelchOI" + str(iterations))
        bitext = self.bitext
        N, self.targetLengthSet = self.maxTargetSentenceLength(bitext)

        logger.info("N " + str(N))
        indexMap, biword = self.mapBitextToInt(self.fe_count)

        sd_size = len(indexMap)
        totalGammaDeltaOAO_t_i = None
        totalGammaDeltaOAO_t_overall_states_over_dest = None

        twoN = 2 * N

        self.a = [[[0.0 for x in range(N + 1)]
                  for y in range(twoN + 1)]
                  for z in range(twoN + 1)]
        self.pi = [0.0 for x in range(twoN + 1)]

        for iteration in range(iterations):
            logger.info("HMMBWTypeI Iteration " + str(iteration))

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
            for (f, e) in bitext:
                self.task.progress("BaumWelch iter %d, %d of %d" %
                                   (iteration, counter, len(bitext),))
                counter += 1
                c = defaultdict(float)

                if iteration == 0:
                    self.initialiseModel(len(e))

                for i in range(len(f)):
                    for j in range(len(e)):
                        small_t[i][j] = self.t[(f[i], e[j])]

                alpha_hat, c_scaled = self.forwardWithTScaled(f, e, small_t)
                beta_hat = self.backwardWithTScaled(f, e, c_scaled, small_t)

                # Setting gamma
                for t in range(1, len(f) + 1):
                    logLikelihood += -1 * log(c_scaled[t])
                    for i in range(1, len(e) + 1):
                        gamma[i][t] =\
                            (alpha_hat[i][t] * beta_hat[i][t]) / c_scaled[t]

                        totalGammaDeltaOAO_t_i[
                            indexMap[(f[t - 1], e[i - 1])]] += gamma[i][t]

                logLikelihood += -1 * log(c_scaled[len(f)])

                for t in range(1, len(f)):
                    for i in range(1, len(e) + 1):
                        for j in range(1, len(e) + 1):
                            c[j - i] += (alpha_hat[i][t] *
                                         self.a[i][j][len(e)] *
                                         small_t[t][j - 1] *
                                         beta_hat[j][t + 1])

                for i in range(1, len(e) + 1):
                    for j in range(1, len(e) + 1):
                        totalC_j_Minus_iOAO[i][j][len(e)] += c[j - i]
                        totalC_l_Minus_iOAO[i][len(e)] += c[j - i]
                    totalGamma1OAO[i] += gamma[i][1]
            # end of loop over bitext

            start_time = time.time()

            logger.info("likelihood " + str(logLikelihood))
            N = len(totalGamma1OAO) - 1

            for k in range(sd_size):
                f, e = biword[k]

                totalGammaDeltaOAO_t_overall_states_over_dest[e] +=\
                    totalGammaDeltaOAO_t_i[k]

            end_time = time.time()

            logger.info("time spent in the end of E-step: " +
                        str(end_time - start_time))
            logger.info("time spent in E-step: " +
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

            logger.info("set " + str(self.targetLengthSet.keys()))
            for I in self.targetLengthSet:
                for i in range(1, I + 1):
                    for j in range(1, I + 1):
                        self.a[i][j][I] = \
                            totalC_j_Minus_iOAO[i][j][I] /\
                            (totalC_l_Minus_iOAO[i][I] + 1e-37)

            for i in range(1, N + 1):
                self.pi[i] = totalGamma1OAO[i] * (1.0 / len(bitext))

            for k in range(sd_size):
                f, e = biword[k]
                self.t[(f, e)] = totalGammaDeltaOAO_t_i[k] / \
                    (totalGammaDeltaOAO_t_overall_states_over_dest[e] + 1e-37)

            end2_time = time.time()
            logger.info("time spent in M-step: " +
                        str(end2_time - end_time))
            logger.info("iteration " + str(iteration) + " completed")

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
        if (f, e) in self.t:
            return self.t[(f, e)]
        if e == "null":
            return self.nullEmissionProb
        return 1.0 / v

    def aProbability(self, iPrime, i, I):
        # p(i|i',I) is smoothed to uniform distribution for now -->
        # p(i|i',I) = 1/I
        # we can make it interpolation form like what Och and Ney did
        if I in self.targetLengthSet:
            return self.a[iPrime][i][I]
        return 1.0 / I

    def logViterbi(self, f, e):
        '''
        This function returns alignment of given sentence in two languages
        @param f: source sentence
        @param e: target sentence
        @return: list of alignment
        '''
        N = len(e)
        twoN = 2 * N
        V = [[0.0 for x in range(len(f))] for y in range(twoN + 1)]
        ptr = [[0 for x in range(len(f))] for y in range(twoN + 1)]
        newd = ["null" for x in range(twoN)]
        for i in range(len(e)):
            newd[i] = e[i]
        for i in range(len(e), twoN):
            newd[i] = "null"

        for q in range(1, twoN + 1):
            tPr = self.tProbability(f[0], newd[q - 1])
            if q >= len(self.pi):
                V[q][0] = - sys.maxint - 1
            elif tPr == 0 or self.pi[q] == 0:
                V[q][0] = - sys.maxint - 1
            else:
                V[q][0] = log(self.pi[q]) + log(tPr)

        for t in range(1, len(f)):
            for q in range(1, twoN + 1):
                maximum = - sys.maxint - 1
                max_q = - sys.maxint - 1
                tPr = self.tProbability(f[t], newd[q - 1])
                for q_prime in range(1, twoN + 1):
                    aPr = self.aProbability(q_prime, q, N)
                    if (aPr != 0) and (tPr != 0):
                        temp = V[q_prime][t - 1] + log(aPr) + log(tPr)
                        if temp > maximum:
                            maximum = temp
                            max_q = q_prime
                V[q][t] = maximum
                ptr[q][t] = max_q

        max_of_V = - sys.maxint - 1
        q_of_max_of_V = 0
        for q in range(1, twoN + 1):
            if V[q][len(f) - 1] > max_of_V:
                max_of_V = V[q][len(f) - 1]
                q_of_max_of_V = q

        trace = []
        trace.append(q_of_max_of_V)
        q = q_of_max_of_V
        i = len(f) - 1
        while (i > 0):
            q = ptr[q][i]
            trace = [q] + trace
            i = i - 1
        return trace

    def decode(self, bitext):
        logger.info("Start decoding")
        logger.info("Testing size: " + str(len(bitext)))
        result = []

        for (f, e) in bitext:
            sentenceAlignment = []
            N = len(e)
            bestAlignment = self.logViterbi(f, e)

            for i in range(len(bestAlignment)):
                if bestAlignment[i] <= len(e):
                    sentenceAlignment.append((i + 1, bestAlignment[i]))

            result.append(sentenceAlignment)
        logger.info("Decoding Completed")
        return result

    def train(self, bitext, iterations):
        self.task = Task("Aligner", "HMMOI" + str(iterations))
        self.task.progress("Training IBM model 1")
        logger.info("Training IBM model 1")
        alignerIBM1 = AlignerIBM1()
        alignerIBM1.train(bitext, iterations)
        self.initWithIBM(alignerIBM1, bitext)
        self.task.progress("IBM model Trained")
        logger.info("IBM model Trained")
        self.baumWelch(iterations=iterations)
        self.task.progress("finalising")
        self.multiplyOneMinusP0H()
        self.task = None
        return
