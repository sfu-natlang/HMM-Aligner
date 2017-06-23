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


class AlignmentModel(IBM1Base):
    def __init__(self):
        IBM1Base.__init__(self)
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate

        self.lambd = 1 - 1e-20
        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15

        self.typeMap = {
            "SEM": 0,
            "FUN": 1,
            "PDE": 2,
            "CDE": 3,
            "MDE": 4,
            "GIS": 5,
            "GIF": 6,
            "COI": 7,
            "TIN": 8,
            "NTR": 9,
            "MTA": 10}
        return

    def initialiseModel(self, tritext):
        IBM1Base.initialiseModel(self, tritext)
        self.total_f_e_h = defaultdict(float)
        self.s = defaultdict(float)

        for (f, e, alignment) in tritext:
            # Initialise total_f_e_h count
            for item in alignment:
                left, right = item.split("-")
                fwords = ''.join(c for c in left if c.isdigit() or c == ',')
                fwords = fwords.split(',')
                if len(fwords) != 1:
                    continue
                # Process source word
                fWord = f[int(fwords[0]) - 1]

                # Process right(target word/types)
                tag = right[len(right) - 4: len(right) - 1]
                tagId = self.typeMap[tag]
                eWords = right[:len(right) - 5]
                eWords = ''.join(c for c in eWords if c.isdigit() or c == ',')
                eWords = eWords.split(',')

                if (eWords[0] != ""):
                    for eStr in eWords:
                        eWord = e[int(eStr) - 1]
                        self.total_f_e_h[(fWord, eWord, tagId)] += 1

        for f, e, h in self.total_f_e_h:
            self.s[str((f, e, h))] =\
                self.total_f_e_h[(f, e, h)] / self.fe_count[(f, e)]
        return



    def train(self, formTritext, tagTritext, iterations=5):
        self.logger.info("Stage 1 Start Training with POS Tags")
