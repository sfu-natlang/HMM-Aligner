# -*- coding: utf-8 -*-

#
# IBM model 1 implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner.
#
import numpy as np
from collections import defaultdict
from loggers import logging
from models.IBM1Base import AlignmentModelBase as IBM1Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(IBM1Base):
    def __init__(self):
        self.modelName = "IBM1"
        self.version = "0.3b"
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

    def _beginningOfIteration(self, index=0):
        self.c = [defaultdict(float) for i in range(len(self.fLex[index]))]
        self.total = [0.0 for i in range(len(self.eLex[index]))]
        return

    def _updateCount(self, f, e, index):
        fLen = len(f)
        eLen = len(e)
        fWords = np.array([f[i][index] for i in range(fLen)])
        eWords = np.array([e[j][index] for j in range(eLen)])
        tSmall = self.tProbability(f, e, index)
        tSmall = tSmall / tSmall.sum(axis=1)[:, None]
        for i in range(fLen):
            tmp = tSmall[i]
            for j in range(eLen):
                self.c[fWords[i]][eWords[j]] += tmp[j]
                self.total[eWords[j]] += tmp[j]
        return

    def _updateEndOfIteration(self, index):
        self.logger.info("End of iteration")
        # Update t
        for i in range(len(self.fLex[index])):
            for j in self.c[i]:
                self.t[i][j] = self.c[i][j] / self.total[j]
        return
