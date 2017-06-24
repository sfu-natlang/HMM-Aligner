# -*- coding: utf-8 -*-

#
# IBM model 1 implementation(old) of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner.
#
import sys
import os
import time
from collections import defaultdict
from loggers import logging
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
        self.t = defaultdict(float)
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate
        Base.__init__(self)
        return

    def initWithBitext(self, bitext):
        self.f_count = defaultdict(int)
        self.e_count = defaultdict(int)
        self.fe_count = defaultdict(int)
        # Initialise f_count
        for (f, e) in bitext:
            for f_i in f:
                self.f_count[f_i] += 1
                # Initialise fe_count
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1

        # Initialise e_count
        for e_j in e:
            self.e_count[e_j] += 1
        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def train(self, bitext, iterations=5):
        task = Task("Aligner", "IBM1OI" + str(iterations))
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(bitext)))
        start_time = time.time()

        self.initWithBitext(bitext)
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
                task.progress("IBM1Old iter %d, %d of %d" %
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
