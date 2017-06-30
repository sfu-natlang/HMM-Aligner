# -*- coding: utf-8 -*-

#
# IBM model 1 base of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the base model for IBM1
#
import time
from copy import deepcopy
from collections import defaultdict
from loggers import logging
from models.modelBase import Task
from models.modelBase import AlignmentModelBase as Base
__version__ = "0.4a"


class AlignmentModelBase(Base):
    def __init__(self):
        self.t = defaultdict(float)
        if "logger" not in vars(self):
            self.logger = logging.getLogger('IBM1BASE')
        if "modelComponents" not in vars(self):
            self.modelComponents = ["t"]
        Base.__init__(self)
        return

    def tProbability(self, f, e, index=0):
        if (f[index], e[index]) in self.t:
            tmp = self.t[(f[index], e[index])]
            if tmp == 0:
                return 0.000006123586217
            else:
                return tmp
        else:
            return 0.000006123586217

    def EM(self, dataset, iterations, modelName="IBM1Base", index=0):
        task = Task("Aligner", modelName + str(iterations))
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(dataset)))
        start_time = time.time()

        for iteration in range(iterations):
            self._beginningOfIteration()
            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for item in dataset:
                f, e = item[0:2]
                counter += 1
                task.progress(modelName + " iter %d, %d of %d" %
                              (iteration, counter, len(dataset),))
                for fWord in f:
                    z = 0
                    for eWord in e:
                        z += self.tProbability(fWord, eWord)
                    for eWord in e:
                        self._updateCount(fWord, eWord, z, index)

            self._updateEndOfIteration()

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        self.endOfEM()
        return

    def decode(self, dataset):
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(dataset)))
        result = []

        for sentence in dataset:

            sentenceAlignment = self.decodeSentence(sentence)

            result.append(sentenceAlignment)
        self.logger.info("Decoding Complete")
        return result

    def decodeSentence(self, sentence):
        # This is the standard sentence decoder for IBM model 1
        # What happens there is that for every source f word, we find the
        # target e word with the highest tr(e|f) score here, which is
        # tProbability(f[i], e[j])
        f, e, alignment = sentence
        sentenceAlignment = []
        for i in range(len(f)):
            max_t = 0
            argmax = -1
            for j in range(len(e)):
                t = self.tProbability(f[i], e[j])
                if t > max_t:
                    max_t = t
                    argmax = j
            sentenceAlignment.append((i + 1, argmax + 1))
        return sentenceAlignment

    def _beginningOfIteration(self):
        '''
        This is the very basic IBM 1 model implementation
        The sample code here is the standard way of updating counts.
        The original equation: tr(e|f) = C(e,f) / C(f)
        Here, C(e, f) is self.c[(f, e)]
              C(f)    is self.total[f]
        As an example, we clear them up at the beginning of every iteration
        here.
        '''

        # self.c = defaultdict(float)
        # self.total = defaultdict(float)
        # return
        raise NotImplementedError

    def _updateCount(self, fWord, eWord, z, index=0):
        '''
        This is the very basic IBM 1 model implementation
        The sample code here is the standard way of updating counts.
        The original equation: tr(e|f) = C(e,f) / C(f)
        Here, tr(e|f) is self.t[(f, e)],
              C(e, f) is self.c[(f, e)]
              C(f)    is self.total[f]
        '''

        # self.c[(fWord[index], eWord[index])] +=\
        #     self.tProbability(fWord[index], eWord[index]) / z
        # self.total[eWord[index]] +=\
        #     self.tProbability(fWord[index], eWord[index]) / z
        raise NotImplementedError

    def _updateEndOfIteration(self):
        '''
        This is where one does smoothing and other stuff.
        The baseline IBM 1 model is very simple and doesn't contain smoothing
        here: tr(e|f) = C(e,f) / C(f)
        Here, tr(e|f) is self.t[(f, e)],
              C(e, f) is self.c[(f, e)]
              C(f)    is self.total[f]
        All of the pairs of words (f, e) can be found in self.fe_count
        '''

        # for (f[0], e[0]) in self.c:
        #     # Change the following line to add smoothing
        #     self.t[(f[0], e[0])] = self.c[(f[0], e[0])] / self.total[e[0]]
        # return
        raise NotImplementedError

    def endOfEM(self):
        '''
        At the end of the EM algorithm, it removes all unnecessary parts of the
        model to save memory space and make it easier for dumping the model.
        '''
        del self.f_count
        del self.e_count
        del self.fe_count
        del self.c
        del self.total
        return
