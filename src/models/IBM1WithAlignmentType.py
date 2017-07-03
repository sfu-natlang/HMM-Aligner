# -*- coding: utf-8 -*-

#
# IBM model 1 with alignment type implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner with alignment type.
#
import numpy as np
from loggers import logging
from models.IBM1Base import AlignmentModelBase as IBM1Base
from evaluators.evaluator import evaluate
__version__ = "0.5a"


class AlignmentModel(IBM1Base):
    def __init__(self):
        self.modelName = "IBM1WithPOSTagAndAlignmentType"
        self.version = "0.3b"
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate
        self.fe = ()

        self.s = np.zeros((0, 0))
        self.sTag = np.zeros((0, 0))
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

    def _beginningOfIteration(self):
        self.c = np.zeros(self.t.shape)
        self.total = np.zeros(self.t.shape[1])
        self.c_feh = np.zeros(self.t.shape + (len(self.typeIndex),))
        return

    def _updateCount(self, fWord, eWord, z, index):
        f, e = fWord[index], eWord[index]
        tPr_z = self.t[f][e] / z
        self.c[f][e] += tPr_z
        self.total[e] += tPr_z
        self.c_feh[fWord[self.index]][eWord[self.index]] +=\
            self.sProbability(fWord, eWord, self.index) * tPr_z
        return

    def _updateEndOfIteration(self):
        self.logger.info("Iteration complete, updating parameters")
        self.t = np.divide(self.c, self.total)
        if self.index == 0:
            del self.s
            self.s = self.keyDiv(self.c_feh, self.c)
        else:
            del self.sTag
            self.sTag = self.keyDiv(self.c_feh, self.c)
        return

    def sProbability(self, f, e, index=0):
        fWord, fTag = f
        eWord, eTag = e
        sTagTmp = (1 - self.lambd) * self.typeDist
        if fTag < self.sTag.shape[0] and eTag < self.sTag.shape[1]:
            sTagTmp += self.sTag[fTag][eTag] * self.lambd
        if index == 1:
            return sTagTmp

        sTmp = (1 - self.lambd) * self.typeDist
        if fWord < self.s.shape[0] and eWord < self.s.shape[1]:
            sTmp += self.s[fWord][eWord] * self.lambd
        return (self.lambda1 * sTmp +
                self.lambda2 * sTagTmp +
                self.lambda3 * self.typeDist)

    def decodeSentence(self, sentence):
        f, e, align = sentence
        sentenceAlignment = []
        for i in range(len(f)):
            max_ts = 0
            argmax = -1
            bestType = -1
            for j in range(len(e)):
                t = 1
                sTmp = self.sProbability(f[i], e[j])
                h = np.argmax(sTmp)
                score = sTmp[h] * t
                if score > max_ts:
                    max_ts = score
                    argmax = j
                    bestType = h
            sentenceAlignment.append(
                (i + 1, argmax + 1, self.typeList[bestType]))
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
