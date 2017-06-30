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
import gzip
import os
import cPickle as pickle
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

    def loadModel(self, fileName=None, force=False):
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
                if force:
                    self.logger.warning("Current model requires file for " +
                                        self.modelName + " model, while the" +
                                        " model of the model file is " +
                                        modelName)
                    self.logger.warning("Under current setting, will force" +
                                        "load. Good luck.")
                else:
                    raise RuntimeError("Current model requires file for " +
                                       self.modelName + " model, while the" +
                                       " model of the model file is " +
                                       modelName)

            if "supportedVersion" in entity:
                if modelVersion not in self.supportedVersion:
                    if force:
                        self.logger.warning(
                            "Unsupported version of model file")
                        self.logger.warning(
                            "Current setting will force load. Good luck.")
                    else:
                        raise RuntimeError("Unsupported version of model file")

        # load components
        for componentName in self.modelComponents:
            if componentName not in entity:
                raise RuntimeError("object " + componentName +
                                   " doesn't exist in this class")
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

    def initialiseBiwordCount(self, dataset, index=0):
        # We don't use .clear() here for reusability of models.
        # Sometimes one would need one or more of the following parts for other
        # Purposes. We wouldn't want to accidentally clear them up.
        self.t = defaultdict(float)
        self.f_count = defaultdict(int)
        self.e_count = defaultdict(int)
        self.fe_count = defaultdict(int)

        for item in dataset:
            f, e = item[0:2]
            for f_i in f:
                self.f_count[f_i[index]] += 1
                for e_j in e:
                    self.fe_count[(f_i[index], e_j[index])] += 1
            for e_j in e:
                self.e_count[e_j[index]] += 1

        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        return

    def initialiseAlignTypeDist(self, dataset, loadTypeDist={}):
        typeDist = defaultdict(float)
        typeTotalCount = 0
        for (f, e, alignment) in dataset:
            # Initialise total_f_e_type count
            for (f_i, e_i, typ) in alignment:
                typeDist[typ] += 1
                typeTotalCount += 1

        # Calculate alignment type distribution
        for typ in typeDist:
            typeDist[typ] /= typeTotalCount
        # Manually override alignment type distribution
        for typ in loadTypeDist:
            typeDist[typ] = loadTypeDist[typ]

        # Create typeIndex and typeList
        self.typeList = []
        self.typeIndex = {}
        for typ in typeDist:
            self.typeList.append(typ)
            self.typeIndex[typ] = len(self.typeList) - 1
        self.typeDist = []
        for h in range(len(self.typeList)):
            self.typeDist.append(typeDist[self.typeList[h]])
        return

    def calculateS(self, dataset, fe_count, index=0):
        total_f_e_type = defaultdict(float)

        for (f, e, alignment) in dataset:
            # Initialise total_f_e_type count
            for (f_i, e_i, typ) in alignment:
                fWord = f[f_i - 1]
                eWord = e[e_i - 1]
                total_f_e_type[(fWord[index],
                                eWord[index],
                                self.typeIndex[typ])] += 1

        s = defaultdict(list)
        for key in fe_count:
            s[key] = [0.0 for h in range(len(self.typeIndex))]
        for f, e, t in total_f_e_type:
            s[(f, e)][t] = total_f_e_type[(f, e, t)] / fe_count[(f, e)]
        return s
