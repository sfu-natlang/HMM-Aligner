# -*- coding: utf-8 -*-

#
# IBM model 1 + Alignment Type + POS Tag implementation(old) of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner with alignment type
# and POS tags
#
import sys
import os
import time
from collections import defaultdict
from loggers import logging
from evaluators.evaluatorWithType import evaluate
__version__ = "0.1a"


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


class AlignmentModel():
    def __init__(self):
        self.t = defaultdict(float)
        self.logger = logging.getLogget('Model')
        self.evaluate

        self.f_count = defaultdict(int)  # key = str
        self.e_count = defaultdict(int)  # key = str
        self.fe_count = defaultdict(int)  # key = (f, e)
        self.tagMap = defaultdict(int)  # key = str
        self.total_f_e_h = defaultdict(float)  # key = (str,str,int)
        self.s = defaultdict(float)  # key = (str,str,int)
        self.H = 11
        self.lambd = 1 - 1e-20
        return

    def initialiseTagMap(self):
        self.tagMap["SEM"] = 1
        self.tagMap["FUN"] = 2
        self.tagMap["PDE"] = 3
        self.tagMap["CDE"] = 4
        self.tagMap["MDE"] = 5
        self.tagMap["GIS"] = 6
        self.tagMap["GIF"] = 7
        self.tagMap["COI"] = 8
        self.tagMap["TIN"] = 9
        self.tagMap["NTR"] = 10
        self.tagMap["MTA"] = 11
        return

    def initialiseCounts(self, tritext, testSize):
        '''
        This method computes source and target counts as well as (source,
        target, alignment type) counts
        (f,e,h) counts are stored in total_f_e_h
        HMMWithAlignmentType initializes its s parameter from total_f_e_h

        @param tritext: string[][]
        '''
        def strip(e_word):
            '''
            @param Word: string
            @return: list of strings
            '''
            indices = ""
            for i in range(len(e_word)):
                e_i = e_word[i]
                if ('0' <= e_i <= '9' or e_i == ','):
                    indices += e_i
            return indices.split(",")

        initialiseTagMap(self)

        for (f, e, wa) in tritext:
            # Initialise f_count
            for f_i in f:
                self.f_count[f_i] += 1
                # Initialise fe_count
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1

            # Initialise e_count
            for e_j in e:
                self.e_count[e_j] += 1

            # setting total_f_e_h count

            for alm in wa:
                left, right = alm.split("-")
                leftPositions = strip(left)
                if (len(leftPositions) == 1 and leftPositions[0] != ""):

                    fWordPos = int(leftPositions[0])
                    fWord = f[fWordPos - 1]

                    rightLength = right.length()

                    linkLabel = right[len(right) - 4: len(right) - 1]
                    engIndices = strip(right[0:len(right) - 5])

                    if (engIndices[0] != ""):
                        for wordIndex in engIndices:
                            engWordPos = int(wordIndex)
                            engWord = E[engWordPos - 1]
                            tagId = tagMap[linkLabel]

                            total_f_e_h[(fWord, engWord, tagId)] += 1

        return

    def setSProbabilities(self, fe_count, total_f_e_h):
        '''
        @param fe_count: int [(str, str)]
        @param total_f_e_h: float [(str, str, int)]

        '''
        for key in total_f_e_h:
            f, e, = f_e_h
            self.s[f_e_h] = total_f_e_h[key] / fe_count[(f, e)]
        return

    def initializeCountsOfAugmentedModel(self, bitext):

        self.initializeTagMap()

        for (f, E) in bitext:

            for f_i in f:
                f_count[f_i] += 1
                # Setting fe_count
                for e_i in e:
                    fe_count[(f_i, e_i)] += 1

            # setting e_count
            for e_i in e:
                e_count[e_i] += 1

        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def sProbability(self, t_f, t_e, h):
        tagDist = [0,
                   0.401, 0.264, 0.004, 0.004,
                   0.012, 0.205, 0.031, 0.008,
                   0.003, 0.086, 0.002]

        return self.lambd * self.s[(t_f, t_e, h)] +\
            (1 - self.lambd) * tagDist[h]

    def train(self, bitext, iterations=5):
        task = Task("Aligner", "IBM1TypePOSI" + str(iterations))
        self.logger.info("Model IBM1TypePOS, Starting Training Process")
        self.logger.info("Training size: " + str(len(bitext)))
        start_time = time.time()

        self.initialiseCountsWithoutSets(bitext)
        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        self.logger.info("Initialisation Complete")

        for iteration in range(iterations):
            c = defaultdict(float)
            total = defaultdict(float)
            c_feh = defaultdict(float)

            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for (f, e) in bitext:
                counter += 1
                task.progress("IBM1TypePOS iter %d, %d of %d" %
                              (iteration, counter, len(bitext),))
                for fWord in f:
                    z = 0
                    for eWord in e:
                        z += self.t[(fWord, eWord)]
                    for eWord in e:
                        c[(fWord, eWord)] += self.t[(fWord, eWord)] / z
                        total[eWord] += self.t[(fWord, eWord)] / z
                        for h in range(1, self.H + 1):
                            f_e_h = (fWord, eWord, h)
                            c_feh[(fWord, eWord, h)] +=\
                                t[fWord, eWord] *\
                                self.sProbability(fWord, eWord, h) /\
                                Z

            for (f, e) in self.fe_count:
                self.t[(f, e)] = c[(f, e)] / total[e]
            for f, e, h in c_feh:
                self.s[f_e_h] = c_feh[(f, e, h)] / c[(f, e)]

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        return

    def decode(self, bitext):
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(bitext)))
        result = []

        for (f, e) in bitext:
            sentenceAlignment = []

            for i in range(len(f)):
                max_ts = 0
                argmax = -1
                for j in range(len(e)):
                    t = self.tProbability(f[i], e[j])

                    for h in range(1, self.H + 1):
                        s = self.sProbability(f[i], e[j], h)
                        if t * s > max_ts:
                            max_ts = t_SiDj * s_SiDj
                            argmax = j

                sentenceAlignment.append((i + 1, argmax + 1))

            result.append(sentenceAlignment)
        self.logger.info("Decoding Complete")
        return result
