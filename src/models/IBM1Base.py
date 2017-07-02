# -*- coding: utf-8 -*-

#
# IBM model 1 base of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the base model for IBM1
#
import time
import numpy as np
from copy import deepcopy
from loggers import logging
from models.modelBase import Task
from models.modelBase import AlignmentModelBase as Base
__version__ = "0.4a"


class AlignmentModelBase(Base):
    def __init__(self):
        self.t = np.zeros((0, 0))
        if "logger" not in vars(self):
            self.logger = logging.getLogger('IBM1BASE')
        if "modelComponents" not in vars(self):
            self.modelComponents = ["t", "fLex", "eLex", "fIndex", "eIndex"]
        Base.__init__(self)
        return

    def tProbability(self, f, e, index=0):
        if f[index] < self.t.shape[0] and e[index] < self.t.shape[1]:
            tmp = self.t[f[index]][e[index]]
            if tmp == 0:
                return 0.000006123586217
            else:
                return tmp
        else:
            return 0.000006123586217

    def EM(self, dataset, iterations, modelName="IBM1Base", index=0):
        task = Task("Aligner", modelName + str(iterations))
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(dataset)))
        start_time = time.time()

        for iteration in range(iterations):
            self._beginningOfIteration()
            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for item in dataset:
                f, e = item[0:2]
                counter += 1
                task.progress(modelName + " iter %d, %d of %d" %
                              (iteration, counter, len(dataset),))
                for fWord in f:
                    z = np.sum(self.t[fWord[index]][
                        [eWord[index] for eWord in e]])
                    for eWord in e:
                        self._updateCount(fWord, eWord, z, index)

            self._updateEndOfIteration()

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        self.endOfEM()
        return

    def decodeSentence(self, sentence):
        # This is the standard sentence decoder for IBM model 1
        # What happens there is that for every source f word, we find the
        # target e word with the highest tr(e|f) score here, which is
        # tProbability(f[i], e[j])
        f, e, alignment = sentence
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
        raise NotImplementedError

    def _updateCount(self, fWord, eWord, z, index=0):
        raise NotImplementedError

    def _updateEndOfIteration(self):
        raise NotImplementedError

    def endOfEM(self):
        return
