#!/usr/bin/env python
import optparse
import sys
import os
import logging
from collections import defaultdict


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
        sys.stderr.write("IBM1 [INFO]: Starting Training Process\n")

        self.initWithBiText(biText)
        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        sys.stderr.write("IBM1 [INFO]: Initialisation Complete\n")

        for iteration in range(iterations):
            c = defaultdict(float)
            total = defaultdict(float)
            sys.stderr.write("IBM1 [INFO]: Starting Iteration " + str(iteration) + "\n")

            for (f, e) in biText:
                for fWord in f:
                    z = 0
                    for eWord in e:
                        z += self.t[(fWord, eWord)]
                    for eWord in e:
                        c[(fWord, eWord)] += self.t[(fWord, eWord)] / z
                        total[eWord] += self.t[(fWord, eWord)] / z

            for (f, e) in self.fe_count:
                self.t[(f, e)] = c[(f, e)] / total[e]

        return

    def decodeToFile(self, biText, fileName):
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
        return

    def decodeToStdout(self, biText):
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
        return


if __name__ == '__main__':
    sys.stderr.write("IBM Model 1 Main Programme\n")
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir", default="sample-data", help="data directory (default=data)")
    optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-v", "--num_tests", dest="num_tests", default=1000, type="int", help="Number of sentences to use for testing")
    optparser.add_option("-i", "--iterations", dest="iter", default=5, type="int", help="Number of iterations to train")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    biText = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    biText2 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_tests]]

    aligner = AlignerIBM1()
    aligner.train(biText, opts.iter)
    # aligner.decodeToStdout(biText2)
    aligner.decodeToFile(biText2, "output_jetic_IBM1")
