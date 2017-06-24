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
        pklFile = gzip.open(fileName, 'rb')
        loadedComponents = pickle.load(pklFile)
        pklFile.close()
        # check model name and version
        if "modelName" in vars(self):
            if "modelName" not in loadedComponents:
                raise RuntimeError("Current model requires model file for " +
                                   self.modelName + " model, while the model" +
                                   " of the model file is not specified.")
            if loadedComponents["modelName"] != self.modelName:
                raise RuntimeError("Current model requires model file for " +
                                   self.modelName + " model, while the model" +
                                   " of the model file is " +
                                   loadedComponents["modelName"])
            if "supportedVersion" in vars(self) and "version" in vars(self):
                if "version" not in loadedComponents:
                    raise RuntimeError("Unsupported version of model file")
                if loadedComponents["version"] not in self.supportedVersion:
                    raise RuntimeError("Unsupported version of model file")

        entity = vars(self)
        for componentName in self.modelComponents:
            if componentName not in entity:
                raise RuntimeError("object in _savedModelFile doesn't exist" +
                                   " in specified model file")
            entity[componentName] = loadedComponents[componentName]

        msg = ""
        if "modelName" in loadedComponents:
            msg += loadedComponents["modelName"]
        if "version" in loadedComponents:
            msg += " v" + loadedComponents["version"]
        self.logger.info("Model loaded: " + msg)
        return

    def saveModel(self, fileName=""):
        if fileName == "":
            self.logger.warning("Destination not specified, model will not" +
                                " be saved")
            return
        self.logger.info("Saving model to " + fileName)
        entity = vars(self)
        component = {}
        # check model name and version
        if "modelName" in vars(self):
            component["modelName"] = self.modelName
            if "version" in vars(self):
                component["version"] = self.version
        for componentName in self.modelComponents:
            if componentName not in entity:
                raise RuntimeError("object in _savedModelFile doesn't exist")
            component[componentName] = entity[componentName]
        output = gzip.open(fileName, 'wb')
        pickle.dump(component, output)
        output.close()
        self.logger.info("Model saved")
        return
