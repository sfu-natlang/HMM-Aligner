# -*- coding: utf-8 -*-

#
# IBM model 1 base of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the base model for IBM1
#
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
from loggers import logging
from models.modelBase import AlignmentModelBase as Base
from evaluators.evaluator import evaluate
__version__ = "0.5a"

tTable = []


def tProbability(f, e, index):
    t = np.zeros((len(f), len(e)))
    for j in range(len(e)):
        for i in range(len(f)):
            t[i][j] = tTable[f[i][index]][e[j][index]]
    t[t == 0] = 0.000006123586217
    return t


def mapFunc(dataset):
    result = defaultdict(float)
    index = 0
    for item in dataset:
        f, e = item[0:2]
        fLen = len(f)
        eLen = len(e)
        fWords = np.array([f[i][index] for i in range(fLen)])
        eWords = np.array([e[j][index] for j in range(eLen)])
        tSmall = tProbability(f, e, index)
        tSmall = tSmall / tSmall.sum(axis=1)[:, None]
        for i in range(fLen):
            tmp = tSmall[i]
            for j in range(eLen):
                result[(f[i][index], e[j][index])] += tmp[j]
                result[e[j][index]] += tmp[j]
    return result.items()


def partition(dataset, size):
    result = []
    slices = (len(dataset) + size - 1) / size
    print size, slices
    for i in range(slices):
        if (i + 1) * size < len(dataset):
            result.append(dataset[i * size: (i + 1) * size])
        else:
            result.append(dataset[i * size:])
    return result


def reduceFunc(item):
    word, occurances = item
    return (word, sum(occurances))


class AlignmentModel(Base):
    def __init__(self):
        self.t = []
        self.modelName = "IBM1"
        self.version = "0.4b"
        self.logger = logging.getLogger('IBM1')
        self.evaluate = evaluate
        self.fLex = self.eLex = self.fIndex = self.eIndex = None
        Base.__init__(self)
        return

    def train(self, dataset, iterations=5):
        dataset = self.initialiseLexikon(dataset)
        self.initialiseBiwordCount(dataset)
        self.EM(dataset, iterations, 'IBM1')
        return

    def tProbability(self, f, e, index=0):
        t = np.zeros((len(f), len(e)))
        for j in range(len(e)):
            if e[j][index] >= len(self.eLex[index]):
                continue
            for i in range(len(f)):
                if f[i][index] < len(self.t) and \
                        e[j][index] in self.t[f[i][index]]:
                    t[i][j] = self.t[f[i][index]][e[j][index]]
        t[t == 0] = 0.000006123586217
        return t

    def EM(self, dataset, iterations, modelName="IBM1Base", index=0):
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(dataset)))

        import multiprocessing
        import itertools
        start_time = time.time()

        for iteration in range(iterations):
            global tTable
            tTable = self.t

            self._beginningOfIteration(index)
            self.logger.info("Starting Iteration " + str(iteration))

            pool = multiprocessing.Pool(None)
            self.logger.info("Map")
            mapResponses =\
                pool.map(mapFunc,
                         partition(dataset, len(dataset) / 4),
                         chunksize=1)
            partitionedData = defaultdict(list)
            for key, value in itertools.chain(*mapResponses):
                partitionedData[key].append(value)
            partitionedData = partitionedData.items()
            self.logger.info("Reduce")
            reducedValues = pool.map(reduceFunc, partitionedData)
            for item, value in reducedValues:
                if isinstance(item, tuple):
                    self.c[item[0]][item[1]] = value
                else:
                    self.total[item] = value

            self._updateEndOfIteration(index)

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        self.endOfEM()

    def decodeSentence(self, sentence):
        f, e, alignment = self.lexiSentence(sentence)
        sentenceAlignment = []
        score = self.tProbability(f, e)
        for i in range(len(f)):
            jBest = np.argmax(score[i])
            sentenceAlignment.append((i + 1, jBest + 1))
        return sentenceAlignment, score

    def _beginningOfIteration(self, index=0):
        self.c = [defaultdict(float) for i in range(len(self.fLex[index]))]
        self.total = [0.0 for i in range(len(self.eLex[index]))]
        return

    def _updateEndOfIteration(self, index):
        self.logger.info("End of iteration")
        # Update t
        for i in range(len(self.fLex[index])):
            for j in self.c[i]:
                self.t[i][j] = self.c[i][j] / self.total[j]
        return

    def endOfEM(self):
        return
