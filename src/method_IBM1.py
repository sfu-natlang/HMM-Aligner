#!/usr/bin/env python
import optparse
import sys
import os
import time
from collections import defaultdict
from loggers import logging, init_logger

if __name__ == "__main__":
    try:
        from progress import Task
    except all:
        Task = None
    init_logger('aligner_jetic_IBM1.log')
logger = logging.getLogger('IBM1')


class AlignerIBM1():
    def __init__(self):
        self.t = defaultdict(float)
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
        logger.info("Starting Training Process")
        logger.info("Training size: " + str(len(biText)))
        start_time = time.time()
        self.initWithBiText(biText)
        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        logger.info("Initialisation Complete")

        for iteration in range(iterations):
            c = defaultdict(float)
            total = defaultdict(float)
            logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for (f, e) in biText:
                counter += 1
                if Task:
                    task.progress("iter %d, %d of %d" % (iteration, counter, len(biText),))
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
        logger.info("Training Complete, total time(seconds): %f\n" % (end_time - start_time,))
        return

    def decodeToFile(self, biText, fileName):
        logger.info("Start decoding to file")
        logger.info("Testing size: " + str(len(biText)))

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
        logger.info("Decoding Complete")
        return

    def decodeToStdout(self, biText):
        logger.info("Start decoding to stdout")
        logger.info("Testing size: " + str(len(biText)))
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
        logger.info("Decoding Complete")
        return


if __name__ == '__main__':
    __logger = logging.getLogger('MAIN')
    __logger.info("IBM Model 1 Main Programme")
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir", default="~/Daten/align-data/", help="data directory (default=~/Daten/align-data/)")
    optparser.add_option("--train", dest="trainData", default="train.20k.seg.cln", help="prefix of parallel data files (default=hansards)")
    optparser.add_option("--test",  dest="testData",  default="test.seg.cln", help="prefix of parallel data files (default=hansards)")

    optparser.add_option("--source", dest="source", default="cn", help="suffix of source language (default=cn)")
    optparser.add_option("--target", dest="target", default="en", help="suffix of target language (default=en)")

    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--trainSize", dest="trainSize", default=20357, type="int", help="Number of sentences to use for training")
    optparser.add_option("-v", "--testSize",  dest="testSize",  default=1956,  type="int", help="Number of sentences to use for testing")
    optparser.add_option("-i", "--iterations", dest="iter", default=5, type="int", help="Number of iterations to train")
    (opts, _) = optparser.parse_args()
    if Task:
        task = Task("JIBM", "T" + str(opts.trainSize) + "D" + str(opts.testSize))
    train_source_data = os.path.expanduser("%s.%s" % (os.path.join(opts.datadir, opts.trainData), opts.source))
    train_target_data = os.path.expanduser("%s.%s" % (os.path.join(opts.datadir, opts.trainData), opts.target))

    test_source_data = os.path.expanduser("%s.%s" % (os.path.join(opts.datadir, opts.testData), opts.source))
    test_target_data = os.path.expanduser("%s.%s" % (os.path.join(opts.datadir, opts.testData), opts.target))

    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    train_biText = [[sentence.strip().split() for sentence in pair] for pair in zip(open(train_source_data), open(train_target_data))[:opts.trainSize]]
    test_biText  = [[sentence.strip().split() for sentence in pair] for pair in zip(open(test_source_data),  open(test_target_data))[:opts.testSize]]

    aligner = AlignerIBM1()
    aligner.train(train_biText, opts.iter)
    # aligner.decodeToStdout(biText2)
    aligner.decodeToFile(test_biText, "output_jetic_IBM1")
