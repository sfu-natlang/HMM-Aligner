# -*- coding: utf-8 -*-

#
# HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the main programme of the HMM aligner
#
import sys
import os
import importlib
import argparse
import multiprocessing
from configparser import SafeConfigParser
from loggers import logging, init_logger
from models.modelChecker import checkAlignmentModel
from fileIO import loadDataset, exportToFile, loadAlignment
__version__ = "0.6a"


if __name__ == '__main__':
    # Default values:
    config = {
        'dataDir': '',
        'sourceLanguage': '',
        'targetLanguage': '',

        'trainData': '',
        'trainDataTag': '',
        'trainAlignment': '',

        'testData': '',
        'testDataTag': '',
        'reference': '',

        'trainSize': sys.maxsize,
        'testSize': sys.maxsize,
        'iterations': 5,
        'model': "cHMM",
        'output': 'o.wa',
        'showFigure': 0,
        'intersect': False,

        'loadModel': "",
        'saveModel': "",
        'forceLoad': False
    }

    configFileDataSection = {
        'DataDirectory': 'dataDir',
        'TargetLanguageSuffix': 'targetLanguage',
        'SourceLanguageSuffix': 'sourceLanguage',
    }

    configFileTrainSection = {
        'TextFilePrefix': 'trainData',
        'TagFilePrefix': 'trainDataTag',
        'AlignmentFileSuffix': 'trainAlignment'
    }

    configFileTestSection = {
        'TextFilePrefix': 'testData',
        'TagFilePrefix': 'testDataTag',
        'Reference': 'reference'
    }

    # Initialise logger
    init_logger('aligner.log')
    __logger = logging.getLogger('MAIN')

    # Dealing with arguments here
    if True:  # Adding arguments
        # Parsing the options
        ap = argparse.ArgumentParser(
            description="""SFU HMM Aligner %s""" % __version__)
        ap.add_argument(
            "-d", "--datadir", dest="dataDir",
            help="data directory")
        ap.add_argument(
            "--train", dest="trainData",
            help="prefix of training data file")
        ap.add_argument(
            "--test", dest="testData",
            help="prefix of testing data file")
        ap.add_argument(
            "--train-tag", dest="trainDataTag",
            help="prefix of training tag file")
        ap.add_argument(
            "--test-tag", dest="testDataTag",
            help="prefix of testing tag file")
        ap.add_argument(
            "--source", dest="sourceLanguage",
            help="suffix of source language")
        ap.add_argument(
            "--target", dest="targetLanguage",
            help="suffix of target language")
        ap.add_argument(
            "-a", "--alignment", dest="trainAlignment",
            help="suffix of alignment file")
        ap.add_argument(
            "-n", "--trainSize", dest="trainSize", type=int,
            help="Number of sentences to use for training")
        ap.add_argument(
            "-v", "--testSize", dest="testSize", type=int,
            help="Number of sentences to use for testing")
        ap.add_argument(
            "-i", "--iterations", dest="iterations", type=int,
            help="Number of iterations to train")
        ap.add_argument(
            "-m", "--model", dest="model",
            help="model to use, default is IBM1Old")
        ap.add_argument(
            "-r", "--reference", dest="reference",
            help="Location of reference file")
        ap.add_argument(
            "-o", "--outputToFile", dest="output",
            help="Path to output file")
        ap.add_argument(
            "-c", "--config", dest="config",
            help="Path to config file")
        ap.add_argument(
            "-s", "--saveModel", dest="saveModel",
            help="Where to save the model")
        ap.add_argument(
            "-l", "--loadModel", dest="loadModel",
            help="Specify the model file to load")
        ap.add_argument(
            "--forceLoad", dest="forceLoad", action='store_true',
            help="Ignore version and force loading model file")
        ap.add_argument(
            "--showFigure", dest="showFigure", type=int,
            help="Show figures for the first specified number of decodings")
        ap.add_argument(
            "--intersect", dest="intersect", action='store_true',
            help="Do intersection training.")
        args = ap.parse_args()

    # Process config file
    if args.config:
        # Check local config path
        if not os.path.isfile(args.config):
            __logger.error("The config file doesn't exist: %s\n" % args.config)
            sys.exit(1)

        # Initialise the config parser
        __logger.info("Reading configurations from file: %s" % (args.config))
        cp = SafeConfigParser(os.environ)
        cp.read(args.config)

        # Process the contents of config file
        for key in configFileDataSection:
            if cp.get('General', key) != '':
                config[configFileDataSection[key]] = cp.get('General', key)

        for key in configFileTrainSection:
            if cp.get('TrainData', key) != '':
                config[configFileTrainSection[key]] = cp.get('TrainData', key)

        for key in configFileTestSection:
            if cp.get('TestData', key) != '':
                config[configFileTestSection[key]] = cp.get('TestData', key)

    # Reset default values to config file
    ap.set_defaults(**config)
    args = ap.parse_args()
    config.update(vars(args))

    # Load model
    __logger.info("Loading model: " + config['model'])
    Model = importlib.import_module("models." + config['model']).AlignmentModel
    if not checkAlignmentModel(Model):
        raise TypeError("Invalid Model class")

    aligner = Model()
    if "version" in vars(aligner):
        __logger.info("Model version: " + str(aligner.version))
    if config['intersect'] is True:
        alignerReverse = Model()

    # Load datasets
    if config['trainData'] != "":
        trainSourceFiles = [os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['trainData']),
                       config['sourceLanguage']))]
        trainTargetFiles = [os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['trainData']),
                       config['targetLanguage']))]
        if config['trainDataTag'] != '':
            trainSourceFiles.append(os.path.expanduser("%s.%s" % (
                os.path.join(config['dataDir'], config['trainDataTag']),
                config['sourceLanguage'])))
            trainTargetFiles.append(os.path.expanduser("%s.%s" % (
                os.path.join(config['dataDir'], config['trainDataTag']),
                config['targetLanguage'])))

        if config['trainAlignment'] != '':
            trainAlignment = os.path.expanduser("%s.%s" % (
                os.path.join(config['dataDir'], config['trainData']),
                config['trainAlignment']))
        else:
            trainAlignment = ''
        __logger.info("Loading dataset")
        trainDataset = loadDataset(trainSourceFiles,
                                   trainTargetFiles,
                                   trainAlignment,
                                   linesToLoad=config['trainSize'])
        if config['intersect'] is True:
            __logger.info("Loading reversed dataset")
            trainDataset2 = loadDataset(trainTargetFiles,
                                        trainSourceFiles,
                                        trainAlignment,
                                        reverse=True,
                                        linesToLoad=config['trainSize'])
        else:
            trainDataset2 = None
    else:
        trainDataset = trainDataset2 = None

    if config['testData'] != "":
        testSourceFiles = [os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['testData']),
                       config['sourceLanguage']))]
        testTargetFiles = [os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['testData']),
                       config['targetLanguage']))]
        if config['testDataTag'] != '':
            testSourceFiles.append(os.path.expanduser("%s.%s" % (
                os.path.join(config['dataDir'], config['testDataTag']),
                config['sourceLanguage'])))
            testTargetFiles.append(os.path.expanduser("%s.%s" % (
                os.path.join(config['dataDir'], config['testDataTag']),
                config['targetLanguage'])))
        testDataset = loadDataset(testSourceFiles, testTargetFiles,
                                  linesToLoad=config['testSize'])
        if config['intersect'] is True:
            testDataset2 = loadDataset(testTargetFiles, testSourceFiles,
                                       linesToLoad=config['testSize'])
        else:
            testDataset2 = None
    else:
        testDataset = testDataset2 = None

    def work(arguments):
        trainDataset, testDataset, reversed = arguments
        aligner = Model()

        if config['loadModel'] != "":
            loadFile = config['loadModel']
            if reversed:
                loadFile = ".".join(loadFile.split(".")[:-1] + ["rev"] +
                                    [loadFile.split(".")[-1]])
            aligner.loadModel(loadFile, force=config['forceLoad'])

        if trainDataset is not None:
            aligner.train(trainDataset, config['iterations'])

        if config['saveModel'] != "":
            saveFile = config['saveModel']
            if reversed:
                if saveFile.endswith("pklz") or saveFile.endswith("pkl"):
                    saveFile = ".".join(saveFile.split(".")[:-1] + ["rev"] +
                                        [saveFile.split(".")[-1]])
                else:
                    saveFile += ".rev"
            aligner.saveModel(saveFile)

        if testDataset is not None:
            alignResult = aligner.decode(testDataset, config['showFigure'])
            return (reversed, alignResult)
        return (None, None)

    arg = [(trainDataset, testDataset, False), ]
    if config['intersect'] is True:
        arg.append((trainDataset2, testDataset2, True))

    if len(arg) == 1:
        result = [work(arg[0])]
    else:
        result = multiprocessing.Pool(2).map(work, arg)

    if config['testData'] != "":
        for (resultReversed, resultAlignment) in result:
            if resultReversed:
                alignResultRev = resultAlignment
            else:
                alignResult = resultAlignment

        if config['intersect'] is True:
            # Intersection is performed here.
            result = []
            for align, alignRev in zip(alignResult, alignResultRev):
                sentenceAlignment = []
                for item in align:
                    if len(item) == 2:
                        # Without alignment type
                        if (item[1], item[0]) in alignRev:
                            sentenceAlignment.append(item)
                    else:
                        # With alignment type
                        if (item[1], item[0], item[2]) in alignRev:
                            sentenceAlignment.append(item)
                result.append(sentenceAlignment)
            alignResult = result

        if config['output'] != "":
            exportToFile(alignResult, config['output'])

        if config['reference'] != "":
            reference = loadAlignment(config['reference'])
            if aligner.evaluate:
                aligner.evaluate(alignResult, reference, config['showFigure'])
        if config['showFigure'] > 0:
            from models.plot import showPlot
            showPlot()
