# -*- coding: utf-8 -*-

#
# IBM model 1 with alignment type implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner with alignment type.
#
from collections import defaultdict
from loggers import logging
from models.IBM1Base import AlignmentModelBase as IBM1Base
from evaluators.evaluator import evaluate
__version__ = "0.4a"


class AlignmentModel(IBM1Base):
    def __init__(self):
        self.modelName = "IBM1WithPOSTagAndAlignmentType"
        self.version = "0.2b"
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate
        self.fe = ()

        self.s = defaultdict(list)
        self.sTag = defaultdict(list)
        self.index = 0
        self.typeList = []
        self.typeIndex = {}
        self.typeDist = []
        self.lambd = 1 - 1e-20
        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15

        self.loadTypeDist = {"SEM": .401, "FUN": .264, "PDE": .004,
                             "CDE": .004, "MDE": .012, "GIS": .205,
                             "GIF": .031, "COI": .008, "TIN": .003,
                             "NTR": .086, "MTA": .002}

        self.modelComponents = ["t", "s", "sTag",
                                "typeList", "typeIndex", "typeDist",
                                "lambd", "lambda1", "lambda2", "lambda3"]
        IBM1Base.__init__(self)
        return

    def _beginningOfIteration(self):
        self.c = defaultdict(float)
        self.total = defaultdict(float)
        self.c_feh = defaultdict(
            lambda: [0.0 for h in range(len(self.typeList))])
        return

    def _updateCount(self, fWord, eWord, z, index):
        tPr_z = self.tProbability(fWord, eWord) / z
        self.c[(fWord[self.index], eWord[self.index])] += tPr_z
        self.total[eWord[self.index]] += tPr_z
        c_feh = self.c_feh[(fWord[self.index], eWord[self.index])]
        for h in range(len(self.typeIndex)):
            c_feh[h] += tPr_z * self.sProbability(fWord, eWord, h)
        return

    def _updateEndOfIteration(self):
        for (f, e) in self.c:
            self.t[(f, e)] = self.c[(f, e)] / self.total[e]
        s = self.s if self.index == 0 else self.sTag
        for f, e in self.c_feh:
            c_feh = self.c_feh[(f, e)]
            s_tmp = s[(f, e)]
            for h in range(len(self.typeIndex)):
                s_tmp[h] = c_feh[h] / self.c[(f, e)]
        return

    def sProbability(self, f, e, h):
        fWord, fTag = f
        eWord, eTag = e
        if self.fe != (f, e):
            self.fe, sKey, sTagKey = (f, e), (f[0], e[0]), (f[1], e[1])
            self.sTmp = self.s[sKey] if sKey in self.s else None
            self.sTagTmp = self.sTag[sTagKey] if sTagKey in self.sTag else None
        sTmp = self.sTmp[h] if self.sTmp else 0
        sTagTmp = self.sTagTmp[h] if self.sTagTmp else 0
        if self.index == 0:
            p1 = (1 - self.lambd) * self.typeDist[h] + self.lambd * sTmp
            p2 = (1 - self.lambd) * self.typeDist[h] + self.lambd * sTagTmp
            p3 = self.typeDist[h]
            return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3 * p3
        else:
            return (1 - self.lambd) * self.typeDist[h] + self.lambd * sTagTmp

    def tProbability(self, f, e):
        return IBM1Base.tProbability(self, f, e, self.index)

    def decodeSentence(self, sentence):
        # This is the standard sentence decoder for IBM model 1
        # What happens there is that for every source f word, we find the
        # target e word with the highest tr(e|f) score here, which is
        # tProbability(f[i], e[j])
        f, e, decodeSentence = sentence
        sentenceAlignment = []
        for i in range(len(f)):
            max_ts = 0
            argmax = -1
            bestType = -1
            for j in range(len(e)):
                t = self.tProbability(f, e)
                for h in range(len(self.typeIndex)):
                    s = self.sProbability(f[i], e[j], h)
                    if t * s > max_ts:
                        max_ts = t * s
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
        self.sTag = self.calculateS(dataset, self.fe_count, self.index)
        self.logger.info("Initialisation complete")
        self.EM(dataset, iterations, 'IBM1TypeS1')
        # reset self.index to 0
        self.index = 0
        self.logger.info("Stage 1 Complete")
        return

    def trainStage2(self, dataset, iterations=5):
        self.logger.info("Stage 2 Start Training with FORM")
        self.logger.info("Initialising model with FORM")
        self.initialiseBiwordCount(dataset, self.index)
        self.s = self.calculateS(dataset, self.fe_count, self.index)
        self.logger.info("Initialisation complete")
        self.EM(dataset, iterations, 'IBM1TypeS2')
        self.logger.info("Stage 2 Complete")
        return

    def train(self, dataset, iterations=5):
        self.logger.info("Initialising Alignment Type Distribution")
        self.initialiseAlignTypeDist(dataset, self.loadTypeDist)
        self.trainStage1(dataset, iterations)
        self.trainStage2(dataset, iterations)
        return
