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
import cython
from copy import deepcopy
from loggers import logging
from models.cModelBase import AlignmentModelBase as Base
__version__ = "0.4a"


@cython.boundscheck(False)
class AlignmentModelBase(Base):
    def __init__(self):
        self.t = []
        if "logger" not in vars(self):
            self.logger = logging.getLogger('IBM1BASE')
        if "modelComponents" not in vars(self):
            self.modelComponents = ["t", "fLex", "eLex", "fIndex", "eIndex"]
        Base.__init__(self)
        return

    def tProbability(self, f, e, index=0):
        cdef int fLen = len(f)
        cdef int eLen = len(e)
        cdef double[:,:] t = np.full((fLen, eLen), 0.000006123586217)
        for i in range(fLen):
            if f[i][index] > len(self.t):
                continue
            tTmp = self.t[f[i][index]]
            for j in range(eLen):
                if e[j][index] in tTmp:
                    t[i][j] = tTmp[e[j][index]]
        return np.array(t)

    def EM(self, dataset, iterations, index=0):
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(dataset)))
        start_time = time.time()

        for iteration in range(iterations):
            self._beginningOfIteration(index)
            self.logger.info("Starting Iteration " + str(iteration))

            for item in dataset:

                self._updateCount(item[0], item[1], index)

            self._updateEndOfIteration(index)

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        self.endOfEM()
        return

    def decodeSentence(self, sentence):
        # This is the standard sentence decoder for IBM model 1
        # What happens there is that for every source f word, we find the
        # target e word with the highest tr(f|e) score here.
        f, e, alignment = self.lexiSentence(sentence)
        sentenceAlignment = []
        score = self.tProbability(f, e)
        for i in range(len(f)):
            jBest = np.argmax(score[i])
            sentenceAlignment.append((i + 1, jBest + 1))
        return sentenceAlignment, score

    def _beginningOfIteration(self, index):
        raise NotImplementedError

    def _updateCount(self, fWord, eWord, z, index=0):
        raise NotImplementedError

    def _updateEndOfIteration(self, index):
        raise NotImplementedError

    def endOfEM(self):
        return
