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

        self.tagMap = {
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
        return

    def initialiseCounts(self, bitext):
        self.total_f_e_h = defaultdict(float)
        self.s = defaultdict(float)

        for item in bitext:
            f, e = item[0:2]
            # Initialise f_count, fe_count, e_count
            for f_i in f:
                self.f_count[f_i] += 1
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1
            for e_j in e:
                self.e_count[e_j] += 1

            # Initialise total_f_e_h count if given tritext. This step is
            # important for HMM
            if len(item) > 2:
                alignment = item[2]
            else:
                alignment = []
            for item in alignment:
                left, right = item.split("-")
                fwords = ''.join(c for c in left if c.isdigit() or c == ',')
                fwords = fwords.split(',')
                if len(fwords) != 1:
                    continue
                # Process source word
                fWord = f[int(fwords[0]) - 1]

                # Process right(target word/tags)
                tag = right[len(right) - 4: len(right) - 1]
                tagId = self.tagMap[tag]
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

        self.initialiseCounts(bitext)
        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        self.logger.info("Initialisation Complete")

        for iteration in range(iterations):
            c = defaultdict(float)
            total = defaultdict(float)
            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for item in bitext:
                f, e = item[0:2]
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

        for item in bitext:
            f, e = item[0:2]
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
