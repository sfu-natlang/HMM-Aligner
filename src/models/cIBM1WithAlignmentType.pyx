# -*- coding: utf-8 -*-

#
# IBM model 1 with alignment type implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner with alignment type.
#
import numpy as np
from collections import defaultdict
from loggers import logging
from models.cIBM1Base import AlignmentModelBase as IBM1Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(IBM1Base):
    def __init__(self):
        self.modelName = "IBM1WithPOSTagAndAlignmentType"
        self.version = "0.4b"
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate
        self.fe = ()

        self.s = []
        self.sTag = []
        self.index = 0
        self.typeList = []
        self.typeIndex = {}
        self.typeDist = np.zeros(0)
        self.lambd = 1 - 1e-20
        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15
        self.fLex = self.eLex = self.fIndex = self.eIndex = None

        self.loadTypeDist = {"SEM": .401, "FUN": .264, "PDE": .004,
                             "CDE": .004, "MDE": .012, "GIS": .205,
                             "GIF": .031, "COI": .008, "TIN": .003,
                             "NTR": .086, "MTA": .002}

        self.modelComponents = ["t", "s", "sTag",
                                "fLex", "eLex", "fIndex", "eIndex",
                                "typeList", "typeIndex", "typeDist",
                                "lambd", "lambda1", "lambda2", "lambda3"]
        IBM1Base.__init__(self)
        return

    def _beginningOfIteration(self, index=0):
        self.c = [defaultdict(float) for i in range(len(self.fLex[index]))]
        self.total = [0.0 for i in range(len(self.eLex[index]))]
        self.c_feh = [defaultdict(lambda: np.zeros(len(self.typeIndex)))
                      for i in range(len(self.fLex[index]))]
        return

    def _updateCount(self, f, e, index):
        fLen = len(f)
        eLen = len(e)
        fWords = np.array([f[i][index] for i in range(fLen)])
        eWords = np.array([e[j][index] for j in range(eLen)])
        tSmall = self.tProbability(f, e, index)
        tSmall = tSmall / tSmall.sum(axis=1)[:, None]
        score = self.sProbability(f, e, index) * tSmall[:, :, None]
        for i in range(fLen):
            tmp = tSmall[i]
            tmps = score[i]
            for j in range(eLen):
                self.c[fWords[i]][eWords[j]] += tmp[j]
                self.total[eWords[j]] += tmp[j]
                self.c_feh[fWords[i]][eWords[j]] += tmps[j]
        return

    def _updateEndOfIteration(self, index):
        self.logger.info("Iteration complete, updating parameters")
        # Update t
        for i in range(len(self.fLex[index])):
            for j in self.c[i]:
                self.t[i][j] = self.c[i][j] / self.total[j]

        # Update s
        if self.index == 0:
            del self.s
            self.s = self.c_feh
        else:
            del self.sTag
            self.sTag = self.c_feh
        for i in range(len(self.c_feh)):
            for j in self.c_feh[i]:
                self.c_feh[i][j] /= self.c[i][j]
        return

    def sProbability(self, f, e, index=0):
        sTag = np.tile((1 - self.lambd) * self.typeDist, (len(f), len(e), 1))

        for j in range(len(e)):
            for i in range(len(f)):
                if f[i][1] < len(self.sTag) and e[j][1] in self.sTag[f[i][1]]:
                    sTag[i][j] += self.lambd * self.sTag[f[i][1]][e[j][1]]
        if index == 1:
            return sTag

        s = np.tile((1 - self.lambd) * self.typeDist, (len(f), len(e), 1))
        for j in range(len(e)):
            for i in range(len(f)):
                if f[i][0] < len(self.s) and e[j][0] in self.s[f[i][0]]:
                    s[i][j] += self.lambd * self.s[f[i][0]][e[j][0]]

        return (self.lambda1 * s +
                self.lambda2 * sTag +
                np.tile(self.lambda3 * self.typeDist, (len(f), len(e), 1)))

    def decodeSentence(self, sentence):
        f, e, align = self.lexiSentence(sentence)
        sentenceAlignment = []
        # t = self.tProbability(f, e)
        t = np.full((len(f), len(e)), 1.)
        sPr = self.sProbability(f, e)
        types = np.argmax(sPr, axis=2)
        score = np.max(sPr, axis=2) * t
        for i in range(len(f)):
            jBest = np.argmax(score[i])
            hBest = types[i][jBest]
            sentenceAlignment.append(
                (i + 1, jBest + 1, self.typeList[hBest]))
        return sentenceAlignment

    def trainStage1(self, dataset, iterations=5):
        self.logger.info("Stage 1 Start Training with POS Tags")
        self.logger.info("Initialising model with POS Tags")
        # self.index set to 1 means training with POS Tag
        self.index = 1
        self.initialiseBiwordCount(dataset, self.index)
        self.sTag = self.calculateS(dataset, self.index)
        self.logger.info("Initialisation complete")
        self.EM(dataset, iterations, 'IBM1TypeS1', self.index)
        # reset self.index to 0
        self.index = 0
        self.logger.info("Stage 1 Complete")
        return

    def trainStage2(self, dataset, iterations=5):
        self.logger.info("Stage 2 Start Training with FORM")
        self.logger.info("Initialising model with FORM")
        self.initialiseBiwordCount(dataset, self.index)
        self.s = self.calculateS(dataset, self.index)
        self.logger.info("Initialisation complete")
        self.EM(dataset, iterations, 'IBM1TypeS2', self.index)
        self.logger.info("Stage 2 Complete")
        return

    def train(self, dataset, iterations=5):
        dataset = self.initialiseLexikon(dataset)
        self.logger.info("Initialising Alignment Type Distribution")
        self.initialiseAlignTypeDist(dataset, self.loadTypeDist)
        self.trainStage1(dataset, iterations)
        self.trainStage2(dataset, iterations)
        return
