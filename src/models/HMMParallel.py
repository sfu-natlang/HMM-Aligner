# -*- coding: utf-8 -*-

#
# HMM model base of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the base model for HMM
#
import sys
import time
import numpy as np
import cython
from math import log
from collections import defaultdict
from copy import deepcopy

from loggers import logging
from models.HMM import AlignmentModel as Base
from evaluators.evaluator import evaluate
__version__ = "0.5a"


@cython.boundscheck(False)
def mapFunc(dataset):
    result = []
    index = 0
    model = AlignmentModel()
    for item, pi, a, tSmall in dataset:
        f, e = item[0:2]
        fLen = len(f)
        eLen = len(e)
        fWords = np.array([f[i][index] for i in range(fLen)])
        eWords = np.array([e[j][index] for j in range(eLen)])
        model.pi = pi
        alpha, alphaScale, beta =\
            AlignmentModel.forwardBackward(model, f, e, tSmall, a)
        gamma = ((alpha * beta).T / alphaScale).T
        gamma = ((alpha * beta).T / alphaScale).T
        xi = np.zeros((len(f), len(e), len(e)))
        xi[1:] = alpha[:-1][..., None] * a[1:] *\
            (beta * tSmall)[1:][:, None, :]
        result.append((gamma, xi, f, e))
    return result


class AlignmentModel(Base):
    def baumWelch(self, dataset, iterations=5, index=0):
        self.logger.info("Starting BaumWelch Training Process, size: " +
                         str(len(dataset)))
        startTime = time.time()
        import multiprocessing
        import itertools
        pool = multiprocessing.Pool(None)

        maxE = max([len(e) for (f, e, alignment) in dataset])
        for (f, e, alignment) in dataset:
            self.eLengthSet[len(e)] = 1
        self.initialiseParameter(maxE)
        self.logger.info("Maximum Target sentence length: " + str(maxE))

        for iteration in range(iterations):
            self.logger.info("BaumWelch Iteration " + str(iteration))
            self._beginningOfIteration(dataset, maxE, index)

            self.logger.info("Map")
            # """
            mapResponses = pool.map(mapFunc,
                                    self.partition(dataset,
                                                   len(dataset) / 4,
                                                   iteration),
                                    chunksize=1)
            """
            mapResponses = [mapFunc(self.partition(dataset,
                                                   len(dataset),
                                                   iteration)[0])]
            """

            self.logger.info("Reduce")
            for cake in mapResponses:
                for (gamma, xi, f, e) in cake:
                    self.EStepGamma(f, e, gamma, index)
                    self.EStepDelta(f, e, xi)

            # M-Step
            self.logger.info("End of iteration, M steps")
            self.MStepDelta(maxE, index)
            self.MStepGamma(maxE, index)

        self.logger.info("Finalising")
        self.endOfBaumWelch(index)
        endTime = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (endTime - startTime,))
        return

    def partition(self, dataset, size, iteration):
        result = []
        slices = (len(dataset) + size - 1) / size
        print size, slices
        for i in range(slices):
            if (i + 1) * size < len(dataset):
                result.append(dataset[i * size: (i + 1) * size])
            else:
                result.append(dataset[i * size:])
            for i in range(len(result[-1])):
                sent = result[-1][i]
                if iteration == 0:
                    self.initialValues(len(sent[1]))
                result[-1][i] = (
                    sent,
                    self.pi,
                    self.aProbability(sent[0], sent[1])[
                        :len(sent[0]), :len(sent[1]), :len(sent[1])],
                    self.tProbability(sent[0], sent[1], 0))

        return result
