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
        self.version = "0.1b"
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate
        self.s = defaultdict(float)
        self.sTag = defaultdict(float)
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

    def initialiseModel(self, dataset, loadTypeDist={}):
        # self.index = 0 for text, 1 for POS Tag
        index = self.index
        self.logger.info("Initialising IBM model")
        IBM1Base.initialiseModel(self, dataset, index)
        total_f_e_type = defaultdict(float)
        # if index == 0 initialise self.s, otherwise initialise self.sTag
        if index == 0:
            self.s = defaultdict(float)
            s = self.s
        else:
            self.sTag = defaultdict(float)
            s = self.sTag
        typeDist = defaultdict(float)
        typeTotalCount = 0

        for (f, e, alignment) in dataset:
            # Initialise total_f_e_type count
            for (f_i, e_i, typ) in alignment:
                fWord = f[f_i - 1]
                eWord = e[e_i - 1]
                total_f_e_type[(fWord[index], eWord[index], typ)] += 1
                typeDist[typ] += 1
                typeTotalCount += 1

        # Calculate alignment type distribution
        for typ in typeDist:
            typeDist[typ] /= typeTotalCount
        # Manually override alignment type distribution
        for typ in loadTypeDist:
            typeDist[typ] = loadTypeDist[typ]

        # Create typeIndex and typeList
        self.typeList = []
        self.typeIndex = {}
        for typ in typeDist:
            self.typeList.append(typ)
            self.typeIndex[typ] = len(self.typeList) - 1
        self.typeDist = []
        for h in range(len(self.typeList)):
            self.typeDist.append(typeDist[self.typeList[h]])

        for f, e, typ in total_f_e_type:
            s[(f, e, self.typeIndex[typ])] =\
                total_f_e_type[(f, e, typ)] / self.fe_count[(f, e)]
        return

    def _beginningOfIteration(self):
        self.c = defaultdict(float)
        self.total = defaultdict(float)
        self.c_feh = defaultdict(float)
        return

    def _updateCount(self, fWord, eWord, z):
        self.c[(fWord[self.index], eWord[self.index])] +=\
            self.tProbability(fWord, eWord) / z
        self.total[eWord[self.index]] +=\
            self.tProbability(fWord, eWord) / z
        for h in range(len(self.typeIndex)):
            self.c_feh[(fWord[self.index], eWord[self.index], h)] +=\
                self.tProbability(fWord, eWord) *\
                self.sProbability(fWord, eWord, h) /\
                z
        return

    def _updateEndOfIteration(self):
        for (f, e) in self.c:
            self.t[(f, e)] = self.c[(f, e)] / self.total[e]
        if self.index == 0:
            for f, e, h in self.c_feh:
                self.s[(f, e, h)] =\
                    self.c_feh[(f, e, h)] / self.c[(f, e)]
        else:
            for f, e, h in self.c_feh:
                self.sTag[(f, e, h)] =\
                    self.c_feh[(f, e, h)] / self.c[(f, e)]
        return

    def sProbability(self, f, e, h):
        fWord, fTag = f
        eWord, eTag = e
        if self.index == 0:
            p1 = (1 - self.lambd) * self.typeDist[h] +\
                self.lambd * self.s[(fWord, eWord, h)]
            p2 = (1 - self.lambd) * self.typeDist[h] +\
                self.lambd * self.sTag[(fTag, eTag, h)]
            p3 = self.typeDist[h]

            return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3 * p3
        else:
            return self.lambd * self.sTag[(fTag, eTag, h)] +\
                (1 - self.lambd) * self.typeDist[h]

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

    def train(self, dataset, iterations=5):
        self.logger.info("Stage 1 Start Training with POS Tags")
        self.logger.info("Initialising")

        self.index = 1
        self.initialiseModel(dataset, self.loadTypeDist)
        self.logger.info("Initialisation complete")

        self.EM(dataset, iterations, 'IBM1TypeS1')
        self.logger.info("Stage 1 Complete, preparing for stage 2")

        self.index = 0
        self.logger.info("Stage 2 Start Training with FORM")
        self.logger.info("Initialising")

        self.initialiseModel(dataset, self.loadTypeDist)
        self.logger.info("Initialisation complete")

        self.EM(dataset, iterations, 'IBM1TypeS2')
        self.logger.info("Stage 2 Complete")
        return
