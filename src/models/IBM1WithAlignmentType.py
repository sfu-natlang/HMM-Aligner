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
__version__ = "0.2a"


class AlignmentModel(IBM1Base):
    def __init__(self):
        IBM1Base.__init__(self)
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate

        self.lambd = 1 - 1e-20
        self.lambda1 = 0.9999999999
        self.lambda2 = 9.999900827395436E-11
        self.lambda3 = 1.000000082740371E-15

        self.typeMap = {"SEM": 0, "FUN": 1, "PDE": 2, "CDE": 3,
                        "MDE": 4, "GIS": 5, "GIF": 6, "COI": 7,
                        "TIN": 8, "NTR": 9, "MTA": 10}
        self.typeDist = [0.401, 0.264, 0.004, 0.004,
                         0.012, 0.205, 0.031, 0.008,
                         0.003, 0.086, 0.002]
        self.linkMap = ["SEM", "FUN", "PDE", "CDE",
                        "MDE", "GIS", "GIF", "COI",
                        "TIN", "NTR", "MTA"]
        return

    def initialiseModel(self, tritext):
        IBM1Base.initialiseModel(self, tritext)
        total_f_e_h = defaultdict(float)
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
                        total_f_e_h[(fWord, eWord, tagId)] += 1

        for f, e, h in total_f_e_h:
            self.s[(f, e, h)] =\
                total_f_e_h[(f, e, h)] / self.fe_count[(f, e)]
        return

    def _beginningOfIteration(self):
        self.c = defaultdict(float)
        self.total = defaultdict(float)
        self.c_feh = defaultdict(float)
        return

    def _updateCount(self, fWord, eWord, z):
        self.c[(fWord, eWord)] += self.tProbability(fWord, eWord) / z
        self.total[eWord] += self.tProbability(fWord, eWord) / z
        for h in range(len(self.typeMap)):
            self.c_feh[(fWord, eWord, h)] +=\
                self.tProbability(fWord, eWord) *\
                self.sProbability(fWord, eWord, h) /\
                z
        return

    def _updateCountTag(self, fWord, eWord, z):
        self.c[(fWord[0], eWord[0])] += self.tProbability(fWord, eWord) / z
        self.total[eWord[0]] += self.tProbability(fWord, eWord) / z
        for h in range(len(self.typeMap)):
            self.c_feh[(fWord[0], eWord[0], h)] +=\
                self.tProbability(fWord, eWord) *\
                self.sProbability(fWord, eWord, h) /\
                z
        return

    def _updateEndOfIteration(self):
        for (f, e) in self.fe_count:
            self.t[(f, e)] = self.c[(f, e)] / self.total[e]
        for f, e, h in self.c_feh:
            self.s[(f, e, h)] =\
                self.c_feh[(f, e, h)] / self.c[(f, e)]
        return

    def sProbabilityWithTag(self, f, e, h):
        fWord, fTag = f
        eWord, eTag = e
        p1 = (1 - self.lambd) * self.typeDist[h] +\
            self.lambd * self.s[(fWord, eWord, h)]
        p2 = (1 - self.lambd) * self.typeDist[h] +\
            self.lambd * self.sTag[(fTag, eTag, h)]
        p3 = self.typeDist[h]

        return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3 * p3

    def sProbability(self, f, e, h):
        return self.lambd * self.s[(f, e, h)] +\
            (1 - self.lambd) * self.typeDist[h]

    def tProbabilityWithTag(self, f, e):
        tmp = self.t[(f[0], e[0])]
        if tmp == 0:
            return 0.000006123586217
        else:
            return tmp

    def decodeSentence(self, sentence):
        # This is the standard sentence decoder for IBM model 1
        # What happens there is that for every source f word, we find the
        # target e word with the highest tr(e|f) score here, which is
        # tProbability(f[i], e[j])
        f, e = sentence
        sentenceAlignment = []
        for i in range(len(f)):
            max_ts = 0
            argmax = -1
            bestType = -1
            for j in range(len(e)):
                t = self.tProbability(f[i], e[j])
                for h in range(len(self.typeMap)):
                    s = self.sProbability(f[i], e[j], h)
                    if t * s > max_ts:
                        max_ts = t * s
                        argmax = j
                        bestType = h
            sentenceAlignment.append(
                (i + 1, argmax + 1, self.linkMap[bestType]))
        return sentenceAlignment

    def train(self, formTritext, tagTritext, iterations=5):
        self.logger.info("Stage 1 Start Training with POS Tags")

        self.initialiseModel(tagTritext)
        self.EM(tagTritext, iterations, 'IBM1TypeS1')

        self.sTag = self.s

        self.logger.info("Stage 1 Complete")

        self.tProbability = self.tProbabilityWithTag
        self.sProbability = self.sProbabilityWithTag
        self._updateCount = self._updateCountTag

        tritext = []
        for (f, e, a1), (fTag, eTag, a2) in zip(formTritext, tagTritext):
            tritext.append((zip(f, fTag), zip(e, eTag), a1))

        self.logger.info("Stage 2 Start Training with POS Tags")

        self.initialiseModel(formTritext)
        self.EM(tritext, iterations, 'IBM1TypeS2')

        self.logger.info("Stage 2 Complete")
        return

    def decode(self, formBitext, tagBitext):
        bitext = []
        for form, tag in zip(formBitext, tagBitext):
            f, e = form[0:2]
            fTag, eTag = tag[0:2]
            bitext.append((zip(f, fTag), zip(e, eTag)))

        return IBM1Base.decode(self, bitext)
