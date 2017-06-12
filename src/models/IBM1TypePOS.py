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

# Constants
tagDist = [0,
           0.401, 0.264, 0.004, 0.004,
           0.012, 0.205, 0.031, 0.008,
           0.003, 0.086, 0.002]


class AlignmentModelPOS():
    def __init__(self):
        self.t = defaultdict(float)
        self.logger = logging.getLogger('Model')
        self.evaluate = evaluate

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

    def initialiseWithTritext(self, tritext):
        self.initialiseTagMap()

        for (f, e, alignment) in tritext:
            # Initialise f_count, fe_count, e_count
            for f_i in f:
                self.f_count[f_i] += 1
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1
            for e_j in e:
                self.e_count[e_j] += 1

            # Initialise total_f_e_h count
            for item in alignment:
                left, right = item.split("-")
                fwords = ''.join(c for c in left if c.isdigit() or c == ',')
                if len(fwords) != 1:
                    continue
                # Process source word
                fWord = f[int(fwords[0]) - 1]

                # Process right(target word/tags)
                tag = right[len(right) - 4: len(right) - 1]
                tagId = tagMap[tag]
                eWords = right[:len(right) - 5]
                eWords = ''.join(c for c in eWords if c.isdigit() or c == ',')
                eWords = eWords.split(',')

                if (eWords[0] != ""):
                    for eStr in eWords:
                        eWord = e[int(eStr) - 1]
                        self.total_f_e_h[(fWord, eWord, tagId)] += 1

        for f, e, h in self.total_f_e_h:
            self.s[(f, e, h)] =\
                self.total_f_e_h[(f, e, h)] / self.fe_count[(f, e)]
        return

    def setSProbabilities(self, fe_count, total_f_e_h):
        '''
        This function is not utilised
        @param fe_count: int [(str, str)]
        @param total_f_e_h: float [(str, str, int)]

        '''
        for f, e, h in total_f_e_h:
            self.s[(f, e, h)] = total_f_e_h[(f, e, h)] / fe_count[(f, e)]
        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def sProbability(self, t_f, t_e, h):
        return self.lambd * self.s[(t_f, t_e, h)] +\
            (1 - self.lambd) * tagDist[h]

    def train(self, tritext, iterations=5):
        task = Task("Aligner", "IBM1TypePOSI" + str(iterations))
        self.logger.info("Model IBM1TypePOS, Starting Training Process")
        self.logger.info("Training size: " + str(len(bitext)))
        start_time = time.time()

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

            for (f, e, alignment) in tritext:
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
                            c_feh[(fWord, eWord, h)] +=\
                                self.t[fWord, eWord] *\
                                self.sProbability(fWord, eWord, h) /\
                                z

            for (f, e) in self.fe_count:
                self.t[(f, e)] = c[(f, e)] / total[e]
            for f, e, h in c_feh:
                self.s[(f, e, h)] = c_feh[(f, e, h)] / c[(f, e)]

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


class AlignmentModel():
    def __init__(self):
        self.t = defaultdict(float)
        self.logger = logging.getLogger('Model')
        self.evaluate = evaluate

        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15
        self.bitext_tag_fe = None
        self.sTag = None
        self.t_table = None
        return

        def initialiseCounts(self, tritext, testSize):
            '''
            This method computes source and target counts as well as (source,
            target, alignment type) counts
            (f,e,h) counts are stored in total_f_e_h
            HMMWithAlignmentType initializes its s parameter from total_f_e_h

            @param tritext: string[][]
            '''
            self.initialiseTagMap()

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
                    leftPositions = left.split(",")
                    if len(leftPositions) != 1:
                        continue

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

                            self.total_f_e_h[(fWord, engWord, tagId)] += 1

            for f, e, h in total_f_e_h:
                self.s[(f, e, h)] =\
                    self.total_f_e_h[(f, e, h)] / self.fe_count[(f, e)]
            return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def sProbability(self, fWord, eWord, h, fTag, eTag):
        p1 = (1 - lambda) * tagDist[h] +\
            self.lambd * self.s[(fWord, eWord, h)]
        p2 = (1 - lambda) * tagDist[h] +\
            self.lambd * self.sTag[(fTag, eTag, h)]
        p3 = tagDist[h]

        return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3 * p3

    def train(self, tritext, iterations=5, POStritext):
        POSAligner = AlignmentModelPOS()

        POSAligner.initialiseCounts(POStritext)
        POSAligner.train(tritext)

        self.sTag = POSAligner.s
        self.fe_count = POSAligner.fe_count

        task = Task("Aligner", "IBM1TypeI" + str(iterations))
        self.logger.info("Model IBM1Type, Starting Training Process")
        self.logger.info("Training size: " + str(len(bitext)))
        start_time = time.time()

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

            for (f, e, ) in tritext:
                fTags, eTags, = POStritext[counter]

                counter += 1
                task.progress("IBM1Type iter %d, %d of %d" %
                              (iteration, counter, len(bitext),))

                for fWord, fTag in zip(f, fTags):
                    z = 0
                    for eWord, eTag in zip(e, eTags):
                        z += self.t[(fWord, eWord)]

                    for eWord, eTag in zip(e, eTags):
                        c[(fWord, eWord)] += self.t[(fWord, eWord)] / z
                        total[eWord] += self.t[(fWord, eWord)] / z
                        for h in range(1, self.H + 1):
                            c_feh[(fWord, eWord, h)] +=\
                                self.t[fWord, eWord] / z *\
                                self.sProbability(fWord, eWord, h, fTag, eTag)

            for (f, e) in self.fe_count:
                self.t[(f, e)] = c[(f, e)] / total[e]
            for f, e, h in c_feh:
                self.s[(f, e, h)] = c_feh[(f, e, h)] / c[(f, e)]

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        return
