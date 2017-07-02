# -*- coding: utf-8 -*-

#
# IBM model 1 implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner.
#
import numpy as np
from loggers import logging
from models.IBM1Base import AlignmentModelBase as IBM1Base
from evaluators.evaluator import evaluate
__version__ = "0.5a"


class AlignmentModel(IBM1Base):
    def __init__(self):
        self.modelName = "IBM1"
        self.version = "0.2b"
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate
        self.fLex = self.eLex = self.fIndex = self.eIndex = None

        IBM1Base.__init__(self)
        return

    def train(self, dataset, iterations=5):
        dataset = self.initialiseLexikon(dataset)
        self.initialiseBiwordCount(dataset)
        self.EM(dataset, iterations, 'IBM1')
        return

    def _beginningOfIteration(self):
        self.c = np.zeros(self.t.shape)
        self.total = np.zeros(self.t.shape[1])
        return

    def _updateCount(self, fWord, eWord, z, index=0):
        f, e = fWord[index], eWord[index]
        self.c[f][e] += self.t[f][e] / z
        self.total[e] += self.t[f][e] / z
        return

    def _updateEndOfIteration(self):
        self.t = np.divide(self.c, self.total)
        return
