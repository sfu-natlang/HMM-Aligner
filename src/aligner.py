#!/usr/bin/env python
import optparse
import sys
import os
import importlib
from loggers import logging, init_logger
from models.modelChecker import checkAlignmentModel
from fileIO import loadBitext, loadTritext, exportToFile, loadAlignment
__version__ = "0.2a"


if __name__ == '__main__':
    # Parsing the options
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir",
                         default="~/Daten/align-data/",
                         help="data directory (default=~/Daten/align-data/)")
    optparser.add_option("--train", dest="trainData",
                         default="train.20k.seg.cln",
                         help="prefix of data file(default=train.20k.seg.cln)")
    optparser.add_option("--train-tag", dest="trainTag",
                         default="train.20k.tags",
                         help="prefix of tag file(default=train.20k.tags)")
    optparser.add_option("--test", dest="testData",
                         default="test.seg.cln",
                         help="prefix of data file(default=test.seg.cln)")
    optparser.add_option("--test-tag", dest="testTag",
                         default="test.tags",
                         help="prefix of test tag file(default=test.tags)")
    optparser.add_option("--source", dest="source", default="cn",
                         help="suffix of source language (default=cn)")
    optparser.add_option("--target", dest="target", default="en",
                         help="suffix of target language (default=en)")
    optparser.add_option("--alignment", dest="alignment",
                         default="wa",
                         help="suffix of alignment file(default=wa)")
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
    optparser.add_option("-r", "--reference", dest="reference", default="",
                         help="Location of reference file")
    optparser.add_option("-o", "--outputToFile", dest="output", default="o.wa",
                         help="Path to output file")
    (opts, _) = optparser.parse_args()

    # Initialise logger
    init_logger('aligner.log')
    __logger = logging.getLogger('MAIN')
    __logger.info("Using model: " + opts.model)

    # Load model
    __logger.info("Loading model")
    Model = importlib.import_module("models." + opts.model).AlignmentModel
    modelType = checkAlignmentModel(Model)
    if modelType == -1:
        raise TypeError("Invalid Model class")
    __logger.info("Model loaded")

    aligner = Model()

    if opts.trainData != "":
        trainSource = os.path.expanduser(
            "%s.%s" % (os.path.join(opts.datadir, opts.trainData), opts.source)
        )
        trainTarget = os.path.expanduser(
            "%s.%s" % (os.path.join(opts.datadir, opts.trainData), opts.target)
        )
        if modelType == 1:
            trainBitext = loadBitext(trainSource,
                                     trainTarget,
                                     opts.trainSize)
            aligner.train(trainBitext, opts.iter)

        if modelType == 2:
            trainSourceTag = os.path.expanduser(
                "%s.%s" % (os.path.join(opts.datadir, opts.trainTag),
                           opts.source))
            trainTargetTag = os.path.expanduser(
                "%s.%s" % (os.path.join(opts.datadir, opts.trainTag),
                           opts.target))
            trainAlignment = os.path.expanduser(
                "%s.%s" % (os.path.join(opts.datadir, opts.trainData),
                           opts.alignment))
            trainFormTritext = loadTritext(trainSource,
                                           trainTarget,
                                           trainAlignment,
                                           opts.trainSize)

            trainTagTritext = loadTritext(trainSourceTag,
                                          trainTargetTag,
                                          trainAlignment,
                                          opts.trainSize)

            aligner.train(formTritext=trainFormTritext,
                          tagTritext=trainTagTritext,
                          iterations=opts.iter)

    if opts.testData != "":
        testSource = "%s.%s" %\
            (os.path.join(opts.datadir, opts.testData), opts.source)
        testTarget = "%s.%s" %\
            (os.path.join(opts.datadir, opts.testData), opts.target)

        if modelType == 1:
            testBitext = loadBitext(testSource, testTarget, opts.testSize)
            alignResult = aligner.decode(testBitext)
        if modelType == 2:
            testSourceTag = os.path.expanduser(
                "%s.%s" % (os.path.join(opts.datadir, opts.testTag),
                           opts.source))
            testTargetTag = os.path.expanduser(
                "%s.%s" % (os.path.join(opts.datadir, opts.testTag),
                           opts.target))
            testAlignment = os.path.expanduser(
                "%s.%s" % (os.path.join(opts.datadir, opts.testData),
                           opts.alignment))

            testFormTritext = loadTritext(testSource,
                                          testTarget,
                                          testAlignment,
                                          opts.trainSize)
            testTagTritext = loadTritext(testSourceTag,
                                         testTargetTag,
                                         testAlignment,
                                         opts.trainSize)
            alignResult = aligner.decode(formTritext=testFormTritext,
                                         tagTritext=testTagTritext)

        if opts.output != "":
            exportToFile(alignResult, opts.output)

        if opts.reference != "":
            reference = loadAlignment(opts.reference)
            if aligner.evaluate:
                if modelType == 1:
                    aligner.evaluate(testBitext, alignResult, reference)
                if modelType == 2:
                    aligner.evaluate(testFormTritext, alignResult, reference)
