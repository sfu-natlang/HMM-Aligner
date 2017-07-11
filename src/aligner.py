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
import StringIO
from ConfigParser import SafeConfigParser
from loggers import logging, init_logger
from models.modelChecker import checkAlignmentModel
from fileIO import loadDataset, exportToFile, loadAlignment
__version__ = "0.5a"


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

        'trainSize': sys.maxint,
        'testSize': sys.maxint,
        'iterations': 5,
        'model': "IBM1",
        'output': 'o.wa',

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
    __logger.info("Model loaded")

    if config['loadModel'] != "":
        aligner.loadModel(config['loadModel'], force=config['forceLoad'])

    elif config['trainData'] != "":
        trainSource = os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['trainData']),
                       config['sourceLanguage'])
        )
        trainTarget = os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['trainData']),
                       config['targetLanguage'])
        )
        trainSourceFiles = [trainSource]
        trainTargetFiles = [trainTarget]

        if config['trainDataTag'] != '':
            trainSourceTag = os.path.expanduser(
                "%s.%s" % (os.path.join(config['dataDir'],
                                        config['trainDataTag']
                                        ), config['sourceLanguage'])
            )
            trainTargetTag = os.path.expanduser(
                "%s.%s" % (os.path.join(config['dataDir'],
                                        config['trainDataTag']
                                        ), config['targetLanguage'])
            )
            trainSourceFiles = [trainSource, trainSourceTag]
            trainTargetFiles = [trainTarget, trainTargetTag]

        if config['trainAlignment'] != '':
            trainAlignment = os.path.expanduser(
                "%s.%s" % (os.path.join(config['dataDir'],
                                        config['trainData']
                                        ), config['trainAlignment'])
            )
        else:
            trainAlignment = ''
        trainDataset = loadDataset(trainSourceFiles,
                                   trainTargetFiles,
                                   trainAlignment,
                                   linesToLoad=config['trainSize'])
        aligner.train(trainDataset, config['iterations'])

    else:
        aligner.loadModel(aligner._savedModelFile)

    if config['saveModel'] != "":
        aligner.saveModel(config['saveModel'])

    if config['testData'] != "":
        testSource = os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['testData']),
                       config['sourceLanguage'])
        )
        testTarget = os.path.expanduser(
            "%s.%s" % (os.path.join(config['dataDir'], config['testData']),
                       config['targetLanguage'])
        )
        testSourceFiles = [testSource]
        testTargetFiles = [testTarget]
        if config['testDataTag'] != '':
            testSourceTag = os.path.expanduser(
                "%s.%s" % (os.path.join(config['dataDir'],
                                        config['testDataTag']
                                        ), config['sourceLanguage'])
            )
            testTargetTag = os.path.expanduser(
                "%s.%s" % (os.path.join(config['dataDir'],
                                        config['testDataTag']
                                        ), config['targetLanguage'])
            )
            testSourceFiles = [testSource, testSourceTag]
            testTargetFiles = [testTarget, testTargetTag]

        testDataset = loadDataset(testSourceFiles,
                                  testTargetFiles,
                                  linesToLoad=config['testSize'])
        alignResult = aligner.decode(testDataset)

        if config['output'] != "":
            exportToFile(alignResult, config['output'])

        if config['reference'] != "":
            reference = loadAlignment(config['reference'])
            if aligner.evaluate:
                aligner.evaluate(alignResult, reference)
