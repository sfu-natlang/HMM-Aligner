# -*- coding: utf-8 -*-

#
# IBM model 1 implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner.
#
from collections import defaultdict
from loggers import logging
from models.IBM1Base import AlignmentModelBase as IBM1Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(IBM1Base):
    def __init__(self):
        self.modelName = "IBM1WithPOSTagAndAlignmentType"
        self.version = "0.1b"
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate

        IBM1Base.__init__(self)
        return

    def train(self, bitext, iterations=5):
        self.initialiseModel(bitext)
        self.EM(bitext, iterations, 'IBM1')
        return

    def _beginningOfIteration(self):
        self.c = defaultdict(float)
        self.total = defaultdict(float)
        return

    def _updateCount(self, fWord, eWord, z):
        self.c[(fWord, eWord)] += self.tProbability(fWord, eWord) / z
        self.total[eWord] += self.tProbability(fWord, eWord) / z
        return

    def _updateEndOfIteration(self):
        for (f, e) in self.fe_count:
            self.t[(f, e)] = self.c[(f, e)] / self.total[e]
        return
