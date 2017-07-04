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
__version__ = "0.4a"


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

    def _updateCount(self, f, e, index):
        fLen = len(f)
        eLen = len(e)
        fWords = np.array([f[i][index] for i in range(fLen)])
        eWords = np.array([e[j][index] for j in range(eLen)])
        eDupli = (eWords[:, np.newaxis] == eWords).sum(axis=0)
        tSmall = self.t[fWords][:, eWords]
        for i in range(fLen):
            tmp = tSmall[i] / np.sum(tSmall[i]) * eDupli
            self.c[fWords[i], eWords] += tmp
            self.total[eWords] += tmp
        return

    def _updateEndOfIteration(self):
        self.t = np.divide(self.c, self.total)
        return
