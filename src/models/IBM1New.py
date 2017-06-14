# -*- coding: utf-8 -*-

#
# IBM model 1 implementation(New) of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the new implementation of IBM model 1 word aligner, which added some
# additional method which are not useful.
#
import time
from copy import deepcopy
from collections import defaultdict
from loggers import logging
from evaluators.evaluator import evaluate
from data.Pair import Pair, IntPair
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
        '''
        @var self.f_count: integer defaultdict with string as index
        @var self.e_count: integer defaultdict with string as index
        @var self.fe_count: integer defaultdict with (str, str) as index
        @var self.tagMap: integer defaultdict with string as index
        @var self.total_f_e_h: float defaultdict with (str, str, int) as index
        '''
        self.t = defaultdict(float)
        self.logger = logging.getLogger('IBM1')
        self.f_count = defaultdict(int)
        self.e_count = defaultdict(int)
        self.fe_count = defaultdict(int)

        self.tagMap = defaultdict(int)
        self.total_f_e_h = defaultdict(float)
        self.evaluate = evaluate
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

    def initialiseCountsWithoutSets(self, bitext):
        '''
        Initialises source count, target count and source-target count tables
        (maps)
        @param bitext: bitext of source-target
        '''
        self.initialiseTagMap()

        for (f, e) in bitext:
            # Initialise f_count
            for f_i in f:
                self.f_count[f_i] += 1
                # Initialise fe_count
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1

            # Initialise e_count
            for e_j in e:
                self.e_count[e_j] += 1
        return

    def initialiseCounts(self, tritext, testSize):
        '''
        This method computes source and target counts as well as (source,
        target, alignment type) counts
        (f,e,h) counts are stored in total_f_e_h
        HMMWithAlignmentType initializes its s parameter from total_f_e_h

        @param tritext: string[][]
        @param testSize: int
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
        sentenceNumber = 1
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
            if (sentenceNumber > len(tritext)):
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

            sentenceNumber += 1

        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def train(self, bitext, iterations=5):
        task = Task("Aligner", "IBM1NI" + str(iterations))
        self.logger.info("Starting Training Process")
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
            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for (f, e) in bitext:
                counter += 1
                task.progress("IBM1New iter %d, %d of %d" %
                              (iteration, counter, len(bitext),))
                for fWord in f:
                    z = 0
                    for eWord in e:
                        z += self.t[(fWord, eWord)]
                    for eWord in e:
                        c[(fWord, eWord)] += self.t[(fWord, eWord)] / z
                        total[eWord] += self.t[(fWord, eWord)] / z

            for (f, e) in self.fe_count:
                self.t[(f, e)] = c[(f, e)] / total[e]

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
                max_t = 0
                argmax = -1
                for j in range(len(e)):
                    t = self.tProbability(f[i], e[j])
                    if t > max_t:
                        max_t = t
                        argmax = j
                sentenceAlignment.append((i + 1, argmax + 1))

            result.append(sentenceAlignment)
        self.logger.info("Decoding Complete")
        return result
