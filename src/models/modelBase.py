# -*- coding: utf-8 -*-

#
# Base model of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the base model for all models. It is recommanded that one uses an
# AlignmentModelBase as the parent class of their own model. The base model
# here provides the function to export and load existing models.
#
import os
import sys
import inspect
import gzip
import time
import unittest
import numpy as np
import cPickle as pickle
from copy import deepcopy
from collections import defaultdict
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from loggers import logging
__version__ = "0.4a"


# This is a private module for transmitting test results. Please ignore.
class DummyTask():
    def __init__(self, taskName="Untitled", serial="XXXX"):
        return

    def progress(self, msg):
        return


try:
    from progress import Task
except ImportError:
    Task = DummyTask


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

        modelName = self.__loadObjectFromFile(pklFile)
        modelVersion = self.__loadObjectFromFile(pklFile)
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
            entity[componentName] = self.__loadObjectFromFile(pklFile)

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
            self.__saveObjectToFile(self.modelName, output)
        else:
            self.__saveObjectToFile("Unspecified Model", output)
        if "version" in vars(self):
            self.__saveObjectToFile(self.version, output)
        else:
            self.__saveObjectToFile("???", output)

        # dump components
        for componentName in self.modelComponents:
            if componentName not in entity:
                raise RuntimeError("object in _savedModelFile doesn't exist")
            self.__saveObjectToFile(entity[componentName], output)

        output.close()
        self.logger.info("Model saved")
        return

    def __loadObjectFromFile(self, pklFile):
        a = pickle.load(pklFile)
        if isinstance(a, dict) and "§§NUMPY§§" in a and a["§§NUMPY§§"] == 0.0:
            self.logger.info("Loading Numpy array, size: " + str(len(a)))
            del a["§§NUMPY§§"]
            maxS = None
            for coordinate in a:
                for i in range(len(coordinate)):
                    if not maxS:
                        maxS = [0 for i in range(len(coordinate))]
                    maxS[i] = max(maxS[i], coordinate[i] + 1)
            result = np.zeros(maxS)
            for coordinate in a:
                result[coordinate] = a[coordinate]
            return result
        return a

    def __saveObjectToFile(self, a, output):
        if isinstance(a, np.ndarray):
            self.logger.info("Dumping Numpy array, size: " + str(a.shape) +
                             ", empty entries: " + str(len(zip(*a.nonzero()))))
            aDict = {"§§NUMPY§§": 0.0}
            for coordinate in zip(*a.nonzero()):
                aDict[coordinate] = a[coordinate]
            a = aDict
            pickle.dump(a, output)
            return
        if isinstance(a, defaultdict):
            # Remove zero valued entries from defaultdict
            self.logger.info(
                "Dumping defaultdict, size pre-trim: " + str(len(a)))
            emptyKeys = [key for key in a if a[key] == 0]
            for key in emptyKeys:
                del a[key]
            self.logger.info(
                "Dumping defaultdict, size after trim: " + str(len(a)))
            pickle.dump(a, output)
            return
        pickle.dump(a, output)
        return

    def initialiseBiwordCount(self, dataset, index=0):
        self.logger.info("Initialising Biword table")
        maxf = len(self.fLex[index])
        maxe = len(self.eLex[index])
        initialValue = 1.0 / maxf
        self.t = [defaultdict(float) for i in range(maxf)]

        for item in dataset:
            f, e = item[0:2]
            for f_i in f:
                for e_j in e:
                    self.t[f_i[index]][e_j[index]] = initialValue
        self.logger.info("Biword table initialised")
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
        self.typeDist = np.zeros(len(self.typeList))
        for h in range(len(self.typeList)):
            self.typeDist[h] = typeDist[self.typeList[h]]
        return

    def calculateS(self, dataset, index=0):
        self.logger.info("Initialising S")
        count = np.zeros((len(self.fLex[index]),
                          len(self.eLex[index]),
                          len(self.typeIndex)))
        feCount = np.zeros((len(self.fLex[index]),
                            len(self.eLex[index])))

        for (f, e, alignment) in dataset:
            for f_i in f:
                for e_j in e:
                    feCount[f_i[index]][e_j[index]] += 1
            # Initialise total_f_e_type count
            for (f_i, e_i, typ) in alignment:
                fWord = f[f_i - 1]
                eWord = e[e_i - 1]
                count[fWord[index]][eWord[index]][self.typeIndex[typ]] += 1

        self.logger.info("Writing S")
        s = self.keyDiv(count, feCount)
        self.logger.info("S computed")
        return s

    def keyDiv(self, x, y):
        if x.shape[:-1] != y.shape:
            raise RuntimeError("Incorrect size")
        if len(x.shape) == 3:
            for i, j in zip(*y.nonzero()):
                x[i][j] /= y[i][j]
        elif len(x.shape) == 2:
            for i, in zip(*y.nonzero()):
                x[i] /= y[i]
        return x

    def decode(self, dataset):
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(dataset)))
        result = []

        startTime = time.time()
        for sentence in dataset:
            sentenceAlignment = self.decodeSentence(sentence)

            result.append(sentenceAlignment)
        endTime = time.time()
        self.logger.info("Decoding Complete, total time: " +
                         str(endTime - startTime) + ", average " +
                         str(len(dataset) / (endTime - startTime)) +
                         " sentences per second")
        return result

    def initialiseLexikon(self, dataset, newDataset=False):
        self.logger.info("Creating lexikon")
        indices = len(dataset[0][0][0])
        self.fLex = [[] for i in range(indices)]
        self.eLex = [[] for i in range(indices)]
        self.fIndex = [{} for i in range(indices)]
        self.eIndex = [{} for i in range(indices)]
        for f, e, alignment in dataset:
            for index in range(indices):
                for fWord in f:
                    self.fIndex[index][fWord[index]] = 1
                for eWord in e:
                    self.eIndex[index][eWord[index]] = 1
        if newDataset:
            dataset = deepcopy(dataset)
        self.logger.info("lexikon fsize: " +
                         str([len(f_i) for f_i in self.fIndex]) +
                         "; esize: " + str([len(e_i) for e_i in self.eIndex]))
        for index in range(indices):
            c = 0
            for key in self.fIndex[index]:
                self.fIndex[index][key] = c
                self.fLex[index].append(key)
                c += 1
            c = 0
            for key in self.eIndex[index]:
                self.eIndex[index][key] = c
                self.eLex[index].append(key)
                c += 1
        self.logger.info("Rewriting dataset")
        for f, e, alignment in dataset:
            for i in range(len(f)):
                f[i] = tuple(
                    [self.fIndex[indx][f[i][indx]] for indx in range(indices)])
            for i in range(len(e)):
                e[i] = tuple(
                    [self.eIndex[indx][e[i][indx]] for indx in range(indices)])
        self.logger.info("lexikon created")
        return dataset

    def lexiSentence(self, sentence):
        f, e, alignment = deepcopy(sentence)
        indices = len(self.fIndex)
        for i in range(len(f)):
            f[i] =\
                tuple([self.lexiWord(self.fIndex[index], f[i][index])
                       for index in range(indices)])
        for i in range(len(e)):
            e[i] =\
                tuple([self.lexiWord(self.eIndex[index], e[i][index])
                      for index in range(indices)])
        return f, e, alignment

    def lexiWord(self, lexikon, word):
        if word in lexikon:
            return lexikon[word]
        else:
            return 424242424242

    def sharedLexikon(self, model):
        if not isinstance(model, AlignmentModelBase):
            raise RuntimeError("Invalid object, object must be an instance " +
                               "of a sub class of " +
                               "models.modelBase.AlignmentModelBase")
        self.fLex, self.eLex, self.fIndex, self.eIndex =\
            model.fLex, model.eLex, model.fIndex, model.eIndex


class TestModelBase(unittest.TestCase):
    def testlexiSentence(self):
        model = AlignmentModelBase()
        model.fIndex = [{
            "a": 0,
            "b": 1,
            "c": 2
        }, {
            "d": 3,
            "e": 4,
            "f": 5
        }]
        model.eIndex = [{
            "A": 0,
            "B": 1,
            "C": 2
        }, {
            "D": 3,
            "E": 4,
            "F": 5
        }]
        sentence = (
            [("a", "d"), ("b", "e"), ("c", "f"), ("g", "h")],
            [("A", "D"), ("B", "E"), ("C", "F"), ("G", "H")],
            []
        )
        correct = (
            [(0, 3), (1, 4), (2, 5), (424242424242, 424242424242)],
            [(0, 3), (1, 4), (2, 5), (424242424242, 424242424242)],
            []
        )
        self.assertSequenceEqual(model.lexiSentence(sentence), correct)
        return

    def testKeyDiv3D(self):
        import math
        n = 3
        m = 4
        h = 5
        x = np.arange(n * m * h).reshape((n, m, h)) + 1
        y = np.arange(n * m).reshape(n, m) + 1
        with np.errstate(invalid='ignore', divide='ignore'):
            correct = np.array([[[x[i][j][k] / y[i][j] for k in range(h)]
                                 for j in range(m)]
                                for i in range(n)])
        model = AlignmentModelBase()
        result = model.keyDiv(x, y)
        for i in range(n):
            for j in range(m):
                for k in range(h):
                    self.assertFalse(math.isnan(result[i][j][k]))
                    self.assertEqual(result[i][j][k], correct[i][j][k])
        return

    def testKeyDiv2D(self):
        import math
        n = 3
        m = 4
        x = np.arange(n * m).reshape((n, m)) + 1
        y = np.arange(n) + 1
        with np.errstate(invalid='ignore', divide='ignore'):
            correct = np.array([[x[i][j] / y[i]
                                 for j in range(m)]
                                for i in range(n)])
        model = AlignmentModelBase()
        result = model.keyDiv(x, y)
        for i in range(n):
            for j in range(m):
                    self.assertFalse(math.isnan(result[i][j]))
                    self.assertEqual(result[i][j], correct[i][j])
        return

    def testLoadSaveObjectFromPKL(self):
        testFileName = "support/dump.pkl"
        model = AlignmentModelBase()
        model.t = {("a", "b"): 1,
                   ("c", "d"): 2,
                   ("e", "f"): 3,
                   ("g", "h"): 4}
        model.s = np.arange(27).reshape((3, 3, 3))
        model.sTag = np.arange(9).reshape((3, 3))
        model.modelComponents = ["s", "sTag", "t"]
        model.saveModel(testFileName)

        model2 = AlignmentModelBase()
        model2.t = model2.s = model2.sTag = None
        model2.modelComponents = ["s", "sTag", "t"]
        model2.loadModel(testFileName, True)

        for key in model.t:
            self.assertEqual(model.t[key], model2.t[key])
        self.assertTrue(model.s.all() == model2.s.all())
        self.assertTrue(model.sTag.all() == model2.sTag.all())
        return


if __name__ == '__main__':
    unittest.main()
