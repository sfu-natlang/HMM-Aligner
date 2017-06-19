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
from models.IBM1New import AlignmentModel as AlignerIBM1
from loggers import logging
from evaluators.evaluator import evaluate
logger = logging.getLogger('HMM')


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


class AlignmentModelTag():
    def __init__(self):
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.a = None
        self.pi = None
        self.task = None

        self.lambd = 1 - 1e-20
        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15

        self.typeMap = {
            "SEM": 0,
            "FUN": 1,
            "PDE": 2,
            "CDE": 3,
            "MDE": 4,
            "GIS": 5,
            "GIF": 6,
            "COI": 7,
            "TIN": 8,
            "NTR": 9,
            "MTA": 10
        }
        self.typeDist = [0.401, 0.264, 0.004, 0.004,
                         0.012, 0.205, 0.031, 0.008,
                         0.003, 0.086, 0.002]
        return

    def initWithIBM(self, modelIBM1, tritext):
        self.f_count = modelIBM1.f_count
        self.e_count = modelIBM1.e_count
        self.fe_count = modelIBM1.fe_count
        self.total_f_e_h = modelIBM1.total_f_e_h
        self.t = modelIBM1.t
        self.tritext = tritext
        self.s = defaultdict(float)
        for f, e, h in self.total_f_e_h:
            self.s[(f, e, h)] =\
                self.total_f_e_h[(f, e, h)] /\
                self.fe_count[(f, e)]
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

    def maxTargetSentenceLength(self, tritext):
        maxLength = 0
        targetLengthSet = defaultdict(int)
        for item in tritext:
            f, e = item[0:2]
            tempLength = len(e)
            if tempLength > maxLength:
                maxLength = tempLength
            targetLengthSet[tempLength] += 1
        return (maxLength, targetLengthSet)

    def mapTritextToInt(self, fe_count):
        index = defaultdict(int)
        biword = defaultdict(tuple)
        i = 0
        for key in fe_count:
            index[key] = i
            biword[i] = key
            i += 1
        return (index, biword)

    def baumWelch(self, iterations=5):
        if not self.task:
            self.task = Task("Aligner", "HMMBWTagI" + str(iterations))
        tritext = self.tritext
        N, self.targetLengthSet = self.maxTargetSentenceLength(tritext)

        logger.info("maxTargetSentenceLength(N) " + str(N))
        indexMap, biword = self.mapTritextToInt(self.fe_count)

        L = len(tritext)
        sd_size = len(indexMap)
        totalGammaDeltaOAO_t_i = None
        totalGammaDeltaOAO_t_overall_states_over_dest = None

        twoN = 2 * N

        self.a = [[[0.0 for x in range(N + 1)]
                  for y in range(twoN + 1)]
                  for z in range(twoN + 1)]
        self.pi = [0.0 for x in range(twoN + 1)]

        for iteration in range(iterations):

            logLikelihood = 0

            totalGammaDeltaOAO_t_i = \
                [1e-63 for x in range(sd_size)]
            totalGammaDeltaOAO_t_overall_states_over_dest =\
                defaultdict(float)
            totalGamma1OAO = [0.0 for x in range(N + 1)]
            totalC_j_Minus_iOAO = [[[0.0 for x in range(N + 1)]
                                    for y in range(N + 1)]
                                   for z in range(N + 1)]
            totalC_l_Minus_iOAO = [[0.0 for x in range(N + 1)]
                                   for y in range(N + 1)]
            self.total_f_e_h.clear()

            gamma = \
                [[0.0 for x in range(N * 2 + 1)] for y in range(N + 1)]
            small_t = \
                [[0.0 for x in range(N * 2 + 1)] for y in range(N * 2 + 1)]

            start0_time = time.time()

            counter = 0
            for item in tritext:
                f, e = item[0:2]
                if counter % 100 == 0:
                    logger.info("sentence " + str(counter) +
                                " of iteration " + str(iteration))
                self.task.progress("BaumWelch iter %d, %d of %d" %
                                   (iteration, counter, len(tritext),))
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

                        for h in range(len(self.typeMap)):
                            f_e_h = (f[t - 1], e[i - 1], h)
                            self.total_f_e_h[f_e_h] += \
                                gamma[i][t] * self.sProbability(f[t - 1],
                                                                e[i - 1],
                                                                h)

                # Setting xi(c)
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
            # end of loop over tritext

            start_time = time.time()

            logger.info("likelihood " + str(logLikelihood))
            N = len(totalGamma1OAO) - 1

            for k in range(sd_size):
                totalGammaDeltaOAO_t_i[k] += totalGammaDeltaOAO_t_i[k]
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
            self.a = [[[0.0 for x in range(N + 1)]
                      for y in range(twoN + 1)]
                      for z in range(twoN + 1)]
            self.pi = [0.0 for x in range(twoN + 1)]
            self.t.clear()

            for I in self.targetLengthSet:
                for i in range(1, I + 1):
                    for j in range(1, I + 1):
                        self.a[i][j][I] = \
                            totalC_j_Minus_iOAO[i][j][I] /\
                            (totalC_l_Minus_iOAO[i][I] + 1e-37)

            for i in range(1, N + 1):
                self.pi[i] = totalGamma1OAO[i] * (1.0 / L)

            for k in range(sd_size):
                f, e = biword[k]
                self.t[(f, e)] = totalGammaDeltaOAO_t_i[k] / \
                    (totalGammaDeltaOAO_t_overall_states_over_dest[e] + 1e-37)

            for f, e, h in self.total_f_e_h:
                self.s[(f, e, h)] = (self.total_f_e_h[(f, e, h)] /
                                     totalGammaDeltaOAO_t_i[indexMap[(f, e)]])

            end2_time = time.time()
            logger.info("time spent in M-step: " +
                        str(end2_time - end_time))
            logger.info("iteration " + str(iteration) + " completed")
            self.task = None

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

    def sProbability(self, f, e, h):
        return self.lambd * self.s[(f, e, h)] +\
            (1 - self.lambd) * self.typeDist[h]


class AlignmentModel():
    def __init__(self):
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.a = None
        self.pi = None
        self.task = None
        self.evaluate = evaluate

        self.lambd = 1 - 1e-20
        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15

        self.typeMap = {
            "SEM": 0,
            "FUN": 1,
            "PDE": 2,
            "CDE": 3,
            "MDE": 4,
            "GIS": 5,
            "GIF": 6,
            "COI": 7,
            "TIN": 8,
            "NTR": 9,
            "MTA": 10
        }
        self.typeDist = [0.401, 0.264, 0.004, 0.004,
                         0.012, 0.205, 0.031, 0.008,
                         0.003, 0.086, 0.002]
        return

    def initWithIBM(self, modelIBM1):
        self.f_count = modelIBM1.f_count
        self.e_count = modelIBM1.e_count
        self.fe_count = modelIBM1.fe_count
        self.total_f_e_h = modelIBM1.total_f_e_h
        self.t = modelIBM1.t
        self.s = defaultdict(float)
        for f, e, h in self.total_f_e_h:
            self.s[(f, e, h)] =\
                self.total_f_e_h[(f, e, h)] /\
                self.fe_count[(f, e)]
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
        alphaSum = 1e-63

        for i in range(1, len(e) + 1):
            alpha[i][1] = self.pi[i] * small_t[0][i - 1]
            alphaSum += alpha[i][1]

        alphaScale[1] = 1 / alphaSum
        for i in range(1, len(e) + 1):
            alpha[i][1] *= alphaScale[1]

        for t in range(2, len(f) + 1):
            alphaSum = 1e-63
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

    def maxTargetSentenceLength(self, tritext):
        maxLength = 0
        targetLengthSet = defaultdict(int)
        for item in tritext:
            f, e = item[0:2]
            tempLength = len(e)
            if tempLength > maxLength:
                maxLength = tempLength
            targetLengthSet[tempLength] += 1
        return (maxLength, targetLengthSet)

    def mapTritextToInt(self, fe_count):
        index = defaultdict(int)
        biword = defaultdict(tuple)
        i = 0
        for key in fe_count:
            index[key] = i
            biword[i] = key
            i += 1
        return (index, biword)

    def baumWelch(self, formTritext, tagTritext, iterations=5):
        if not self.task:
            self.task = Task("Aligner", "HMMBWTypeI" + str(iterations))
        N, self.targetLengthSet = self.maxTargetSentenceLength(formTritext)

        logger.info("maxTargetSentenceLength(N) " + str(N))
        indexMap, biword = self.mapTritextToInt(self.fe_count)

        L = len(formTritext)
        sd_size = len(indexMap)
        totalGammaDeltaOAO_t_i = None
        totalGammaDeltaOAO_t_overall_states_over_dest = None

        twoN = 2 * N

        self.a = [[[0.0 for x in range(N + 1)]
                  for y in range(twoN + 1)]
                  for z in range(twoN + 1)]
        self.pi = [0.0 for x in range(twoN + 1)]

        for iteration in range(iterations):

            logLikelihood = 0

            totalGammaDeltaOAO_t_i = \
                [1e-63 for x in range(sd_size)]
            totalGammaDeltaOAO_t_overall_states_over_dest =\
                defaultdict(float)
            totalGamma1OAO = [0.0 for x in range(N + 1)]
            totalC_j_Minus_iOAO = [[[0.0 for x in range(N + 1)]
                                    for y in range(N + 1)]
                                   for z in range(N + 1)]
            totalC_l_Minus_iOAO = [[0.0 for x in range(N + 1)]
                                   for y in range(N + 1)]
            self.total_f_e_h.clear()

            gamma = \
                [[0.0 for x in range(N * 2 + 1)] for y in range(N + 1)]
            small_t = \
                [[0.0 for x in range(N * 2 + 1)] for y in range(N * 2 + 1)]

            start0_time = time.time()

            counter = 0
            for itemForm, itemTag in zip(formTritext, tagTritext):
                f, e = itemTag[0:2]
                fTags, eTags = itemTag[0:2]

                if counter % 100 == 0:
                    logger.info("sentence " + str(counter) +
                                " of iteration " + str(iteration))
                self.task.progress("BaumWelch iter %d, %d of %d" %
                                   (iteration, counter, len(formTritext),))
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

                        for h in range(len(self.typeMap)):
                            f_e_h = (f[t - 1], e[i - 1], h)
                            self.total_f_e_h[f_e_h] += \
                                gamma[i][t] * self.sProbability(f[t - 1],
                                                                e[i - 1],
                                                                h,
                                                                fTags[t - 1],
                                                                eTags[i - 1])

                # Setting xi(c)
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
            # end of loop over tritext

            start_time = time.time()

            logger.info("likelihood " + str(logLikelihood))
            N = len(totalGamma1OAO) - 1

            for k in range(sd_size):
                totalGammaDeltaOAO_t_i[k] += totalGammaDeltaOAO_t_i[k]
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
            self.a = [[[0.0 for x in range(N + 1)]
                      for y in range(twoN + 1)]
                      for z in range(twoN + 1)]
            self.pi = [0.0 for x in range(twoN + 1)]
            self.t.clear()

            for I in self.targetLengthSet:
                for i in range(1, I + 1):
                    for j in range(1, I + 1):
                        self.a[i][j][I] = \
                            totalC_j_Minus_iOAO[i][j][I] /\
                            (totalC_l_Minus_iOAO[i][I] + 1e-37)

            for i in range(1, N + 1):
                self.pi[i] = totalGamma1OAO[i] * (1.0 / L)

            for k in range(sd_size):
                f, e = biword[k]
                self.t[(f, e)] = totalGammaDeltaOAO_t_i[k] / \
                    (totalGammaDeltaOAO_t_overall_states_over_dest[e] + 1e-37)

            for f, e, h in self.total_f_e_h:
                self.s[(f, e, h)] = (self.total_f_e_h[(f, e, h)] /
                                     totalGammaDeltaOAO_t_i[indexMap[(f, e)]])

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

    def sProbability(self, fWord, eWord, h, fTag, eTag):
        first = self.lambd * self.s[(fWord, eWord, h)] +\
            (1 - self.lambd) * self.typeDist[h]
        second = self.lambd * self.sTag[(fTag, eTag, h)] +\
            (1 - self.lambd) * self.typeDist[h]

        return (self.lambda1 * first +
                self.lambda2 * second +
                self.lambda3 * self.typeDist[h])

    def train(self, formTritext, tagTritext, iterations=5):
        self.task = Task("Aligner", "HMMTypeI" + str(iterations))

        self.task.progress("Training IBM model 1 with tags")
        logger.info("Training IBM model 1 with tags")

        alignerIBM1POSTag = AlignerIBM1()
        alignerIBM1POSTag.train(tagTritext, iterations)

        self.task.progress("IBM model with tags Trained")
        logger.info("IBM model with tags Trained")

        self.task.progress("Training HMM with tags")
        logger.info("Training HMM with tags")

        alignerHMMTag = AlignmentModelTag()
        alignerHMMTag.initWithIBM(alignerIBM1POSTag, tagTritext)
        alignerHMMTag.baumWelch(iterations=iterations)
        alignerHMMTag.multiplyOneMinusP0H()
        self.sTag = alignerHMMTag.s

        self.task.progress("HMM model with tags Trained")
        logger.info("HMM model with tags Trained")

        self.task.progress("Training IBM model with FORM")
        logger.info("Training IBM model with FORM")

        alignerIBM1 = AlignerIBM1()
        alignerIBM1.train(formTritext, iterations)

        self.task.progress("IBM model with tags Trained")
        logger.info("IBM model with tags Trained")

        self.task.progress("Training HMM")
        logger.info("Training HMM")

        self.initWithIBM(alignerIBM1)
        self.baumWelch(formTritext, tagTritext, iterations=iterations)
        self.multiplyOneMinusP0H()

        self.task.progress("HMM model Trained")
        logger.info("HMM model with Trained")

        self.task = None
        return

    def logViterbi(self, f, e, fTags, eTags):
        '''
        This function returns alignment of given sentence in two languages
        @param f: source sentence
        @param e: target sentence
        @return: list of alignment
        '''
        N = len(e)
        twoN = 2 * N
        V = [[[0.0 for z in range(len(self.typeMap))]
             for x in range(len(f))]
             for y in range(twoN + 1)]
        ptr = [[[0 for z in range(len(self.typeMap))]
               for x in range(len(f))]
               for y in range(twoN + 1)]
        ptr_h = [[[0 for z in range(len(self.typeMap))]
                 for x in range(len(f))]
                 for y in range(twoN + 1)]
        newd = ["null" for x in range(twoN)]
        newdTags = ["null" for x in range(twoN)]
        for i in range(len(e)):
            newd[i] = e[i]
            newdTags[i] = eTags[i]
        for i in range(len(e), twoN):
            newd[i] = "null"
            newdTags[i] = "null"

        for q in range(1, twoN + 1):
            tPr = self.tProbability(f[0], newd[q - 1])
            for h in range(len(self.typeMap)):
                first = (
                    self.s[(f[0], newd[q - 1], h)] * self.lambd +
                    (1 - self.lambd) * self.typeDist[h])
                second = (
                    self.sTag[(fTags[0]), newdTags[q - 1]] * self.lambd +
                    (1 - self.lambd) * self.typeDist[h])
                s = (self.lambda1 * first +
                     self.lambda2 * second +
                     self.lambda3 * self.typeDist[h])

                if q >= len(self.pi):
                    V[q][0][h] = - sys.maxint - 1
                elif tPr == 0 or self.pi[q] == 0 or s == 0:
                    V[q][0][h] = - sys.maxint - 1
                else:
                    V[q][0][h] = log(self.pi[q]) + log(tPr) + log(s)

        for t in range(1, len(f)):
            for q in range(1, twoN + 1):
                maximum = - sys.maxint - 1
                max_q = - sys.maxint - 1
                max_h = 0
                tPr = self.tProbability(f[t], newd[q - 1])
                for q_prime in range(1, twoN + 1):
                    aPr = self.aProbability(q_prime, q, N)
                    for h in range(len(self.typeMap)):
                        if (aPr != 0) and (tPr != 0):
                            temp = V[q_prime][t - 1][h] + log(aPr) + log(tPr)
                            if temp > maximum:
                                maximum = temp
                                max_q = q_prime
                                max_h = h

                for h in range(len(self.typeMap)):
                    s = self.sProbability(f[t],
                                          newd[q - 1],
                                          h,
                                          fTags[t],
                                          newdTags[q - 1])
                    if s != 0:
                        temp_s = log(s)
                        V[q][t][h] = maximum + temp_s
                        ptr[q][t][h] = max_q
                        ptr_h[q][t][h] = max_h

        max_of_V = - sys.maxint - 1
        q_of_max_of_V = 0
        h_of_max_of_V = 0
        for q in range(1, twoN + 1):
            for h in range(len(self.typeMap)):
                if V[q][len(f) - 1][h] > max_of_V:
                    max_of_V = V[q][len(f) - 1][h]
                    q_of_max_of_V = q
                    h_of_max_of_V = h

        trace = []
        bestLinkTrace = []
        trace.append(q_of_max_of_V)
        bestLinkTrace.append(h_of_max_of_V)

        q = q_of_max_of_V
        i = len(f) - 1
        h = h_of_max_of_V

        while (i > 0):
            qOld = q
            hOld = h

            q = ptr[qOld][i][hOld]
            h = ptr_h[qOld][i][hOld]
            trace = [q] + trace
            bestLinkTrace = [h] + trace
            i = i - 1
        return trace, bestLinkTrace

    def decode(self, formBitext, tagBitext):
        logger.info("Start decoding")
        logger.info("Testing size: " + str(len(formBitext)))
        result = []

        for (f, e), (fTags, eTags) in zip(formBitext, tagBitext):
            sentenceAlignment = []
            N = len(e)
            bestAlign, bestAlignType = self.logViterbi(f, e, fTags, eTags)

            for i in range(len(bestAlign)):
                if bestAlign[i] <= len(e):
                    sentenceAlignment.append(
                        (i + 1, bestAlign[i], bestAlignType[i]))

            result.append(sentenceAlignment)
        logger.info("Decoding Completed")
        return result
