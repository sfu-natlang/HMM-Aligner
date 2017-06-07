#!/usr/bin/env python
import optparse
import sys
import os
import importlib
from loggers import logging, init_logger, task
from models.modelChecker import checkAlignmentModel


class dummyTask():
    def __init__(self, taskName="Untitled", serial="XXXX"):
        return

    def progress(self, msg):
        return


if __name__ == '__main__':
    # Parsing the options
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir",
                         default="~/Daten/align-data/",
                         help="data directory (default=~/Daten/align-data/)")
    optparser.add_option("--train", dest="trainData",
                         default="train.20k.seg.cln",
                         help="prefix of data file(default=train.20k.seg.cln)")
    optparser.add_option("--test", dest="testData",
                         default="test.seg.cln",
                         help="prefix of data file(default=test.seg.cln)")
    optparser.add_option("--source", dest="source", default="cn",
                         help="suffix of source language (default=cn)")
    optparser.add_option("--target", dest="target", default="en",
                         help="suffix of target language (default=en)")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5,
                         type="float",
                         help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--trainSize", dest="trainSize", default=20357,
                         type="int",
                         help="Number of sentences to use for training")
    optparser.add_option("-v", "--testSize", dest="testSize", default=1956,
                         type="int",
                         help="Number of sentences to use for testing")
    optparser.add_option("-i", "--iterations", dest="iter", default=5,
                         type="int", help="Number of iterations to train")
    optparser.add_option("-m", "--model", dest="model", default="IBM1Old",
                         help="model to use, default is IBM1Old")
    (opts, _) = optparser.parse_args()

    # Initialise logger
    init_logger('aligner.log')
    __logger = logging.getLogger('MAIN')
    __logger.info("Using model: " + opts.model)

    # Load model
    __logger.info("Loading model")
    Model = importlib.import_module("models." + opts.model).AlignmentModel
    if not checkAlignmentModel(Model):
        raise TypeError("Invalid Model class")
    __logger.info("Model loaded")

    trainSourcePath = os.path.expanduser(
        "%s.%s" % (os.path.join(opts.datadir, opts.trainData), opts.source))
    trainTargetPath = os.path.expanduser(
        "%s.%s" % (os.path.join(opts.datadir, opts.trainData), opts.target))

    testSourcePath = os.path.expanduser(
        "%s.%s" % (os.path.join(opts.datadir, opts.testData), opts.source))
    testTargetPath = os.path.expanduser(
        "%s.%s" % (os.path.join(opts.datadir, opts.testData), opts.target))

    trainBiText =\
        [[sentence.strip().split() for sentence in pair] for pair in
            zip(open(trainSourcePath), open(trainTargetPath))[:opts.trainSize]]
    testBiText =\
        [[sentence.strip().split() for sentence in pair] for pair in
            zip(open(testSourcePath), open(testTargetPath))[:opts.testSize]]

    aligner = Model()
    aligner.train(trainBiText, opts.iter)
    aligner.decodeToFile(testBiText, "output.wa")
    task.terminate()
