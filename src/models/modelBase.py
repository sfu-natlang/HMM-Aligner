# -*- coding: utf-8 -*-

#
# IBM model 1 base of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the base model for all models. It is recommanded that one uses an
# AlignmentModelBase as the parent class of their own model. The base model
# here provides the function to export and load existing models.
#
import pickle
import gzip
import os
from collections import defaultdict
from loggers import logging
__version__ = "0.4a"


class AlignmentModelBase():
    def __init__(self):
        '''
        self.modelComponents contains all of the names of the variables that
        own wishes to save in the model. It is vital that one calles __init__
        here at the end of their own __init__ as it would try to automatically
        load the model specified in self._savedModelFile.

        One should always include a logger in ones own model. Otherwise a blank
        one will be provided here.

        One should also, always come up with a unique name for their model. It
        will be marked in the saved model files to prevent accidentally loading
        the wrong model. It should be saved in self.modelName.

        Optionally, when there is a self.supportedVersion list and self.version
        str, the loader will only load the files with supported versions.
        '''
        if "logger" not in vars(self):
            self.logger = logging.getLogger('MODEL')
        if "modelComponents" not in vars(self):
            self.modelComponents = []
        if "_savedModelFile" not in vars(self):
            self._savedModelFile = ""
        return

    def loadModel(self, fileName=None):
        if fileName is None:
            fileName = self._savedModelFile
        if fileName == "":
            self.logger.warning("Destination not specified, model will not" +
                                " be loaded")
            return
        self.logger.info("Loading model from " + fileName)
        fileName = os.path.expanduser(fileName)
        if fileName.endswith("pklz"):
            pklFile = gzip.open(fileName, 'rb')
        else:
            pklFile = open(fileName, 'rb')

        modelName = pickle.load(pklFile)
        modelVersion = pickle.load(pklFile)
        if not isinstance(modelName, str) or not isinstance(modelVersion, str):
            raise RuntimeError("Incorrect model file format")

        msg = modelName + " v" + modelVersion
        self.logger.info("model identified as: " + msg)

        entity = vars(self)
        # check model name and version
        if "modelName" in entity:
            if modelName != self.modelName:
                raise RuntimeError("Current model requires model file for " +
                                   self.modelName + " model, while the model" +
                                   " of the model file is " +
                                   modelName)
            if "supportedVersion" in entity:
                if modelVersion not in self.supportedVersion:
                    raise RuntimeError("Unsupported version of model file")

        # load components
        for componentName in self.modelComponents:
            if componentName not in entity:
                raise RuntimeError("object doesn't exist in this class")
            entity[componentName] = pickle.load(pklFile)

        pklFile.close()
        self.logger.info("Model loaded")
        return

    def saveModel(self, fileName=""):
        if fileName == "":
            self.logger.warning("Destination not specified, model will not" +
                                " be saved")
            return
        entity = vars(self)
        if fileName.endswith("pklz"):
            output = gzip.open(fileName, 'wb')
        elif fileName.endswith("pkl"):
            output = open(fileName, 'wb')
        else:
            fileName = fileName + ".pkl"
            output = open(fileName, 'wb')
        self.logger.info("Saving model to " + fileName)

        # dump model name and version
        if "modelName" in vars(self):
            pickle.dump(self.modelName, output)
        else:
            pickle.dump("Unspecified Model", output)
        if "version" in vars(self):
            pickle.dump(self.version, output)
        else:
            pickle.dump("???", output)

        # dump components
        for componentName in self.modelComponents:
            if componentName not in entity:
                raise RuntimeError("object in _savedModelFile doesn't exist")

            # Remove zero valued entires from defaultdict
            if isinstance(entity[componentName], defaultdict):
                self.logger.info("component: " + componentName +
                                 ", size before trim: " +
                                 str(len(entity[componentName])))
                emptyKeys =\
                    [key for key in entity[componentName]
                     if entity[componentName][key] == 0]
                for key in emptyKeys:
                    del entity[componentName][key]
                self.logger.info("component: " + componentName +
                                 ", size after trim: " +
                                 str(len(entity[componentName])))
            pickle.dump(entity[componentName], output)

        output.close()
        self.logger.info("Model saved")
        return
