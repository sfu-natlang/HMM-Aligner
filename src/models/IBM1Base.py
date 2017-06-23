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


class AlignmentModelBase():
    def __init__(self):
        '''
        @var self.f_count: integer defaultdict with string as index
        @var self.e_count: integer defaultdict with string as index
        @var self.fe_count: integer defaultdict with (str, str) as index
        @var self.tagMap: integer defaultdict with string as index
        @var self.total_f_e_h: float defaultdict with (str, str, int) as index
        '''
        self.t = defaultdict(float)
        self.f_count = defaultdict(int)
        self.e_count = defaultdict(int)
        self.fe_count = defaultdict(int)
        self.logger = logging.getLogger('IBM1BASE')
        return

    def initialiseModel(self, bitext):
        # We don't use .clear() here for reusability of models.
        # Sometimes one would need one or more of the following parts for other
        # Purposes. We wouldn't want to accidentally clear them up.
        self.t = defaultdict(float)
        self.f_count = defaultdict(int)
        self.e_count = defaultdict(int)
        self.fe_count = defaultdict(int)

        for item in bitext:
            f, e = item[0:2]
            for f_i in f:
                self.f_count[f_i] += 1
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1
            for e_j in e:
                self.e_count[e_j] += 1

        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def EM(self, bitext, iterations, modelName="IBM1Base"):
        task = Task("Aligner", modelName + str(iterations))
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(bitext)))
        start_time = time.time()

        self.initialiseModel(bitext)

        self.logger.info("Initialisation Complete")

        for iteration in range(iterations):
            self._beginningOfIteration()
            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for item in bitext:
                f, e = item[0:2]
                counter += 1
                task.progress(modelName + " iter %d, %d of %d" %
                              (iteration, counter, len(bitext),))
                for fWord in f:
                    z = 0
                    for eWord in e:
                        z += self.t[(fWord, eWord)]
                    for eWord in e:
                        self._updateCount(fWord, eWord, z)

            self._updateEndOfIteration()

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        return

    def decode(self, bitext):
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(bitext)))
        result = []

        for sentence in bitext:

            sentenceAlignment = self.decodeSentence(sentence)

            result.append(sentenceAlignment)
        self.logger.info("Decoding Complete")
        return result

    def decodeSentence(self, sentence):
        # This is the standard sentence decoder for IBM model 1
        # What happens there is that for every source f word, we find the
        # target e word with the highest tr(e|f) score here, which is
        # tProbability(f[i], e[j])
        f, e = sentence
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
        return sentenceAlignment

    def _beginningOfIteration(self):
        '''
        This is the very basic IBM 1 model implementation
        The sample code here is the standard way of updating counts.
        The original equation: tr(e|f) = C(e,f) / C(f)
        Here, C(e, f) is self.c[(f, e)]
              C(f)    is self.total[f]
        As an example, we clear them up at the beginning of every iteration
        here.
        '''

        # self.c = defaultdict(float)
        # self.total = defaultdict(float)
        # return
        raise NotImplementedError

    def _updateCount(self, fWord, eWord, z):
        '''
        This is the very basic IBM 1 model implementation
        The sample code here is the standard way of updating counts.
        The original equation: tr(e|f) = C(e,f) / C(f)
        Here, tr(e|f) is self.t[(f, e)],
              C(e, f) is self.c[(f, e)]
              C(f)    is self.total[f]
        '''

        # self.c[(fWord, eWord)] += self.t[(fWord, eWord)] / z
        # self.total[eWord] += self.t[(fWord, eWord)] / z
        # return
        raise NotImplementedError

    def _updateEndOfIteration(self):
        '''
        This is where one does smoothing and other stuff.
        The baseline IBM 1 model is very simple and doesn't contain smoothing
        here: tr(e|f) = C(e,f) / C(f)
        Here, tr(e|f) is self.t[(f, e)],
              C(e, f) is self.c[(f, e)]
              C(f)    is self.total[f]
        All of the pairs of words (f, e) can be found in self.fe_count
        '''

        # for (f, e) in self.fe_count:
        #     # Change the following line to add smoothing
        #     self.t[(f, e)] = self.c[(f, e)] / self.total[e]
        # return
        raise NotImplementedError
