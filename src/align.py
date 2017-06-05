#!/usr/bin/env python
import optparse
import sys
import os
import logging
import time
from math import log
from method_IBM1 import AlignerIBM1
from collections import defaultdict


class AlignerHMM():
    def __init__(self):
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        self.a = None
        self.pi = None
        return

    def initWithIBM(self, modelIBM1, biText):
        self.f_count = modelIBM1.f_count
        self.fe_count = modelIBM1.fe_count
        self.t = modelIBM1.t
        self.biText = biText
        sys.stderr.write("HMM [INFO]: IBM Model 1 Loaded\n")
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
                    total += betaHat[j][t + 1] * self.a[i][j][len(e)] * small_t[t][j - 1]
                betaHat[i][t] = alphaScale[t] * total
        return betaHat

    def maxTargetSentenceLength(self, biText):
        maxLength = 0
        targetLengthSet = defaultdict(int)
        for (f, e) in biText:
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
        biText = self.biText
        N, self.targetLengthSet = self.maxTargetSentenceLength(biText)

        sys.stderr.write("HMM [INFO]: N " + str(N) + "\n")
        indexMap, biword = self.mapBitextToInt(self.fe_count)

        L = len(biText)
        sd_size = len(indexMap)
        totalGammaDeltaOverAllObservations_t_i = None
        totalGammaDeltaOverAllObservations_t_overall_states_over_dest = None

        twoN = 2 * N

        self.a = [[[0.0 for x in range(N + 1)] for y in range(twoN + 1)] for z in range(twoN + 1)]
        self.pi = [0.0 for x in range(twoN + 1)]

        for iteration in range(iterations):

            logLikelihood = 0

            totalGammaDeltaOverAllObservations_t_i = [0.0 for x in range(sd_size)]
            totalGammaDeltaOverAllObservations_t_overall_states_over_dest = defaultdict(float)
            totalGamma1OverAllObservations = [0.0 for x in range(N + 1)]
            totalC_j_Minus_iOverAllObservations = [[[0.0 for x in range(N + 1)] for y in range(N + 1)] for z in range(N + 1)]
            totalC_l_Minus_iOverAllObservations = [[0.0 for x in range(N + 1)] for y in range(N + 1)]

            gamma = [[0.0 for x in range(N * 2 + 1)] for y in range(N + 1)]
            small_t = [[0.0 for x in range(N * 2 + 1)] for y in range(N * 2 + 1)]

            start0_time = time.time()

            sent_count = 0
            for (f, e) in biText:
                if sent_count % 100 == 0:
                    sys.stderr.write("HMM [INFO]: sentence: " + str(sent_count) + " of iteration " + str(iteration) + "\n")
                sent_count += 1
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
                        gamma[i][t] = (alpha_hat[i][t] * beta_hat[i][t]) / c_scaled[t]
                        totalGammaDeltaOverAllObservations_t_i[indexMap[(f[t - 1], e[i - 1])]] += gamma[i][t]

                logLikelihood += -1 * log(c_scaled[len(f)])

                for t in range(1, len(f)):
                    for i in range(1, len(e) + 1):
                        for j in range(1, len(e) + 1):
                            c[j - i] += alpha_hat[i][t] * self.a[i][j][len(e)] * small_t[t][j - 1] * beta_hat[j][t + 1]

                for i in range(1, len(e) + 1):
                    for j in range(1, len(e) + 1):
                        totalC_j_Minus_iOverAllObservations[i][j][len(e)] += c[j - i]
                        totalC_l_Minus_iOverAllObservations[i][len(e)] += c[j - i]
                    totalGamma1OverAllObservations[i] += gamma[i][1]
            # end of loop over bitext

            start_time = time.time()

            sys.stderr.write("HMM [INFO]: likelihood " + str(logLikelihood) + "\n")
            N = len(totalGamma1OverAllObservations) - 1

            for k in range(sd_size):
                totalGammaDeltaOverAllObservations_t_i[k] += totalGammaDeltaOverAllObservations_t_i[k]
                f, e = biword[k]
                totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e] += totalGammaDeltaOverAllObservations_t_i[k]

            end_time = time.time()

            sys.stderr.write("HMM [INFO]: time spent in the end of E-step: " + str(end_time - start_time) + "\n")
            sys.stderr.write("HMM [INFO]: time spent in E-step: " + str(end_time - start0_time) + "\n")

            # M-Step
            del self.a
            del self.pi
            del self.t
            self.a = [[[0.0 for x in range(N + 1)] for y in range(twoN + 1)] for z in range(twoN + 1)]
            self.pi = [0.0 for x in range(twoN + 1)]
            self.t = defaultdict(float)

            sys.stderr.write("HMM [INFO]: set " + str(self.targetLengthSet.keys()) + "\n")
            for I in self.targetLengthSet:
                for i in range(1, I + 1):
                    for j in range(1, I + 1):
                        self.a[i][j][I] = totalC_j_Minus_iOverAllObservations[i][j][I] / (totalC_l_Minus_iOverAllObservations[i][I] + 0.0000000000000000000000000000000000001)

            for i in range(1, N + 1):
                self.pi[i] = totalGamma1OverAllObservations[i] * (1.0 / L)

            for k in range(sd_size):
                f, e = biword[k]
                self.t[(f, e)] = totalGammaDeltaOverAllObservations_t_i[k] / (totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e] + 0.0000000000000000000000000000000000001)

            end2_time = time.time()
            sys.stderr.write("HMM [INFO]: time spent in M-step: " + str(end2_time - end_time) + "\n")
            sys.stderr.write("HMM [INFO]: iteration " + str(iteration) + " complete\n")

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
        # p(i|i',I) is smoothed to uniform distribution for now --> p(i|i',I) = 1/I
        # we can make it interpolation form like what Och and Ney did
        if I in self.targetLengthSet:
            return self.a[iPrime][i][I]
        return 1.0 / I

    def logViterbi(self, f, e):
        '''
        This function returns alignment of given sentence in two languages
        param f: source sentence
        param e: target sentence
        return: list of alignment
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
            if tPr == 0 or self.pi[q] == 0:
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

    def findBestAlignmentsForAll_AER(self, biText, fileName):
        outputFile = open(fileName, "w")
        alignmentList = []
        for (f, e) in biText:
            N = len(e)
            bestAlignment = self.logViterbi(f, e)
            line = ""
            for i in range(len(bestAlignment)):
                if bestAlignment[i] <= len(e):
                    line += str(i) + "-" + str(bestAlignment[i] - 1) + " "
            alignmentList.append(line)
            outputFile.write(line + "\n")
            # sys.stdout.write(line + "\n")
        outputFile.close()
        return alignmentList


if __name__ == '__main__':
    sys.stderr.write("HMM Main Programme\n")
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir", default="sample-data", help="data directory (default=data)")
    optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-v", "--num_tests", dest="num_tests", default=1000, type="int", help="Number of sentences to use for testing")
    optparser.add_option("-i", "--iterations", dest="iter", default=5, type="int", help="Number of iterations to train")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    biText = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    biText2 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_tests]]

    alignerIBM1 = AlignerIBM1()
    alignerIBM1.train(biText, opts.iter)
    alignerHMM = AlignerHMM()
    alignerHMM.initWithIBM(alignerIBM1, biText)
    alignerHMM.baumWelch()
    alignerHMM.multiplyOneMinusP0H()
    alignerHMM.findBestAlignmentsForAll_AER(biText2, "output_jetic_HMM")
