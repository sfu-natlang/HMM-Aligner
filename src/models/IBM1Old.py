#!/usr/bin/env python
import optparse
import sys
import os
import time
from collections import defaultdict
from loggers import logging, task


class AlignmentModel():
    def __init__(self):
        self.t = defaultdict(float)
        self.logger = logging.getLogger('IBM1')
        return

    def initWithBiText(self, biText):
        self.f_count = defaultdict(int)
        self.e_count = defaultdict(int)
        self.fe_count = defaultdict(int)
        # Initialise f_count
        for (f, e) in biText:
            for f_i in f:
                self.f_count[f_i] += 1
                # Initialise fe_count
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1

        # Initialise e_count
        for e_j in e:
            self.e_count[e_j] += 1
        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def train(self, biText, iterations=5):
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(biText)))
        start_time = time.time()

        self.initWithBiText(biText)
        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        self.logger.info("Initialisation Complete")

        for iteration in range(iterations):
            c = defaultdict(float)
            total = defaultdict(float)
            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for (f, e) in biText:
                counter += 1
                task.progress("IBM1Old iter %d, %d of %d" %
                              (iteration, counter, len(biText),))
                for fWord in f:
                    z = 0
                    for eWord in e:
                        z += self.t[(fWord, eWord)]
                    for eWord in e:
                        c[(fWord, eWord)] += self.t[(fWord, eWord)] / z
                        total[eWord] += self.t[(fWord, eWord)] / z

            for (f, e) in self.fe_count:
                self.t[(f, e)] = c[(f, e)] / total[e]

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f\n" %
                         (end_time - start_time,))
        return

    def decodeToFile(self, biText, fileName):
        self.logger.info("Start decoding to file")
        self.logger.info("Testing size: " + str(len(biText)))

        outputFile = open(fileName, "w")

        for (f, e) in biText:
            result = []

            for i in range(len(f)):
                max_t = 0
                argmax = -1
                for j in range(len(e)):
                    t = self.tProbability(f[i], e[j])
                    if t > max_t:
                        max_t = t
                        argmax = j
                result.append((i, argmax))

            line = ""
            for (i, j) in result:
                line += str(i) + "-" + str(j) + " "

            outputFile.write(line + "\n")

        outputFile.close()
        self.logger.info("Decoding Complete")
        return

    def decodeToStdout(self, biText):
        self.logger.info("Start decoding to stdout")
        self.logger.info("Testing size: " + str(len(biText)))
        for (f, e) in biText:
            result = []

            for i in range(len(f)):
                max_t = 0
                argmax = -1
                for j in range(len(e)):
                    t = self.tProbability(f[i], e[j])
                    if t > max_t:
                        max_t = t
                        argmax = j
                result.append((i, argmax))

            line = ""
            for (i, j) in result:
                line += str(i) + "-" + str(j) + " "
            sys.stdout.write(line + "\n")
        self.logger.info("Decoding Complete")
        return
