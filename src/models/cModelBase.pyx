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
__version__ = "0.5a"


def isLambda(f):
    lamb = (lambda: 0)
    return isinstance(f, type(lamb)) and f.__name__ == lamb.__name__


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
        if "modelName" not in vars(self):
            self.modelName = "BaseModel"
        if "version" not in vars(self):
            self.version = "0.3b"
        if "logger" not in vars(self):
            self.logger = logging.getLogger('MODEL')
        if "modelComponents" not in vars(self):
            self.modelComponents = []
        if "_savedModelFile" not in vars(self):
            self._savedModelFile = ""
        if "fLex" not in vars(self):
            self.fLex = []
        if "eLex" not in vars(self):
            self.eLex = []
        if "fIndex" not in vars(self):
            self.fIndex = []
        if "eIndex" not in vars(self):
            self.eIndex = []
        return

    def loadModel(self, fileName=None, force=False):
        '''
        This method loads model from specified file. The file will only be
        loaded if it has the right modelName and version. Only components
        listed in a model's modelComponents list will be saved and loaded.
        @param fileName: str. Name of the model file.
        @param force: bool. This option ignores checks on modelName and version
                      Just so that we can all have a happily life let's not use
                      this option.
        @return: Nothing
        '''
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
        '''
        This method saves model to specified file. Only components listed in a
        model's modelComponents list will be saved and loaded.
        @param fileName: str. Name of the model file.
        @return: Nothing
        '''
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
        '''
        This method saves model component to specified file. Why does it exist?
        Well, sometimes a trained model might contain redundant information
        that doesn't need to be saved, and that's where this method comes in.
        @param pklFile: str. Name of the model file.
        @return: loaded component
        '''
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
        '''
        This method saves model component to specified file. Why does it exist?
        Well, sometimes a trained model might contain redundant information
        that doesn't need to be saved, and that's where this method comes in.
        For every component to be saved, this method is called. If you want to
        remove redundant information from a list or a defaultdict, this is the
        perfect place to do just that.
        @param a: object. The model component
        @param output: object. The file(or compressed file).
        @return: loaded component
        '''
        if isinstance(a, np.ndarray):
            self.logger.info("Dumping Numpy array, size: " + str(a.shape) +
                             ", valid entries: " + str(len(zip(*a.nonzero()))))
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
            if isLambda(a.default_factory):
                a.default_factory = float
            pickle.dump(a, output)
            return
        if isinstance(a, list):
            # Remove lambda defaults from defaultdicts in the list
            self.logger.info(
                "Dumping list, size: " + str(len(a)))
            for item in a:
                if isinstance(item, defaultdict) and\
                        isLambda(item.default_factory):
                    item.default_factory = float
            pickle.dump(a, output)
            return
        pickle.dump(a, output)
        return

    def initialiseBiwordCount(self, dataset, index=0):
        '''
        This method initialises the translation probability table by assigning
        every appeared word pair a default value. It can also be used to extend
        an existing translation probability table. Note that no matter what,
        the translation probability table involved here is self.t.

        @param dataset: Dataset. A dataset
        @param index: int. Index indicates which part of the word to work on,
                      by default it's 0 for FORM and 1 for POS Tags.
        @return: Nothing
        '''
        self.logger.info("Initialising Biword table")
        maxf = len(self.fLex[index])
        maxe = len(self.eLex[index])
        initialValue = 1.0 / maxf
        if len(self.t) < maxf:
            newT = [defaultdict(float) for i in range(maxf - len(self.t))]
            self.t += newT

        for item in dataset:
            f, e = item[0:2]
            for f_i in f:
                for e_j in e:
                    if e_j[index] not in self.t[f_i[index]]:
                        self.t[f_i[index]][e_j[index]] = initialValue
        self.logger.info("Biword table initialised")
        return

    def initialiseAlignTypeDist(self, dataset, loadTypeDist={}):
        """
        This is where alignment type distributions and probability are loaded.
        By default the probability is calculated based on annotated data in the
        dataset, but alternatively it can also be maunally overwritten by the
        loadTypeDist option.

        @param dataset: Dataset. A dataset
        @param loadTypeDist: dict. A set of probability of individual alignment
                             types that overwrites the ones calculated.
        @return: Nothing
        """
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

    def calculateS(self, dataset, index=0, oldS=None):
        """
        This is where translation probability with alignment types (S table) is
        initialised. The initialised probability table will be returned. One
        can also extend an existing table with the option oldS.

        @param dataset: Dataset. A dataset
        @param index: int. Index indicates which part of the word to work on,
                      by default it's 0 for FORM and 1 for POS Tags.
        @param oldS: probability table. If oldS is not None, it will be
                     extended and returned.

        @return: The (extended) S table
        """
        self.logger.info("Initialising S")
        count = [defaultdict(lambda: np.zeros(len(self.typeIndex)))
                 for i in range(len(self.fLex[index]))]
        feCount = [defaultdict(float) for i in range(len(self.fLex[index]))]

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
        if oldS:
            if len(oldS) < len(self.fLex[index]):
                newS = [defaultdict(lambda: np.zeros(len(self.typeIndex)))
                        for i in range(len(self.fLex[index]) - len(oldS))]
                oldS += newS
            for i in range(len(count)):
                for j in count[i]:
                    if j not in oldS[i]:
                        oldS[i][j] = count[i][j] / feCount[i][j]
            self.logger.info("S computed")
            return oldS
        for i in range(len(count)):
            for j in count[i]:
                count[i][j] /= feCount[i][j]
        return count

    def keyDiv(self, x, y):
        """
        This method is no longer used in the actual programme.
        """
        if x.shape[:-1] != y.shape:
            raise RuntimeError("Incorrect size")
        if len(x.shape) == 3:
            for i, j in zip(*y.nonzero()):
                x[i][j] /= y[i][j]
        elif len(x.shape) == 2:
            for i, in zip(*y.nonzero()):
                x[i] /= y[i]
        return x

    def decode(self, dataset, showFigure=0):
        """
        This is the decoder. It decodes all sentences in the dataset by calling
        decodeSentence method, which is defined in each models(or modelBases).
        Optionally, it displays scores of alignment by drawing a graph.

        @param dataset: Dataset. A dataset
        @param showFigure: int. Plot the scores of the first specified number
                           of sentences.

        @return: alignment. See API reference for more detail on this structure
        """
        if showFigure > 0:
            from models.plot import plotAlignmentWithScore
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(dataset)))
        result = []
        count = 0

        startTime = time.time()
        for sentence in dataset:
            sentenceAlignment = self.decodeSentence(sentence)
            if len(sentenceAlignment) > 1 and\
                    isinstance(sentenceAlignment[1], np.ndarray):
                sentenceAlignment, score = sentenceAlignment
                if count < showFigure:
                    plotAlignmentWithScore(score,
                                           sentenceAlignment,
                                           f=sentence[0],
                                           e=sentence[1],
                                           # output=str(count))
                                           output=None)
                    count += 1

            result.append(sentenceAlignment)
        endTime = time.time()
        self.logger.info("Decoding Complete, total time: " +
                         str(endTime - startTime) + ", average " +
                         str(len(dataset) / (endTime - startTime)) +
                         " sentences per second")
        return result

    def initialiseLexikon(self, dataset, newDataset=False):
        """
        Create the dictionary. It actually just calls extendLexikon.

        @param dataset: Dataset. A dataset
        @param newDataset: bool. Whether to return a new dataset or just modify
                           the one referenced here.

        @return: A lexicalised dataset.
        """
        self.logger.info("Creating lexikon")
        self.extendLexikon(dataset, newDataset)
        return dataset

    def extendLexikon(self, dataset, newDataset=False):
        """
        Extend the existing dictionary. If there is no dictionary, create one.
        It also lexicalises the dataset. Note that the dataset lexicalised here
        naturally don't contain unknown words, as they are all included in the
        dictionary.

        @param dataset: Dataset. A dataset
        @param newDataset: bool. Whether to return a new dataset or just modify
                           the one referenced here.

        @return: A lexicalised dataset.
        """
        if "fLex" not in vars(self) or self.fLex is None:
            self.fLex, self.eLex, self.fIndex, self.eIndex = [], [], [], []

        indices = len(dataset[0][0][0])
        self.fLex += [[] for i in range(len(self.fLex), indices)]
        self.eLex += [[] for i in range(len(self.eLex), indices)]
        self.fIndex += [{} for i in range(len(self.fIndex), indices)]
        self.eIndex += [{} for i in range(len(self.eIndex), indices)]

        extFIndex = [{} for i in range(indices)]
        extEIndex = [{} for i in range(indices)]

        for f, e, alignment in dataset:
            for index in range(min(indices, len(f[0]))):
                for fWord in f:
                    if fWord[index] not in self.fIndex[index]:
                        extFIndex[index][fWord[index]] = 1
                for eWord in e:
                    if eWord[index] not in self.eIndex[index]:
                        extEIndex[index][eWord[index]] = 1
        if newDataset:
            dataset = deepcopy(dataset)
        self.logger.info("New fWords size: " +
                         str([len(f_i) for f_i in extFIndex]) +
                         "; eWords size: " +
                         str([len(e_i) for e_i in extEIndex]))
        for index in range(indices):
            c = len(self.fLex[index])
            for key in extFIndex[index]:
                self.fIndex[index][key] = c
                self.fLex[index].append(key)
                c += 1
            c = len(self.eLex[index])
            for key in extEIndex[index]:
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
        self.logger.info("lexikon extended")
        return dataset

    def lexiSentence(self, sentence):
        """
        Lexicalise a sentence. Handling of unknown words is defined in lexiWord

        @param sentence: Sentence. A sentence.
        @return: A lexicalised sentence.
        """
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
        """
        Handling unknown words should occur here. If the word is in the lexikon
        then its index is returned. If not, by default it is assigned
        424242424242. Note that the lexikon and the word are from the same
        index. (FORM lexikon for FORMs, TAG lexikon for TAGs)

        @param lexikon: dict. Value for each key is the index of the key.
        @param word: str. The word.
        @return: int. The index of the word.
        """
        if word in lexikon:
            return lexikon[word]
        else:
            return 424242424242

    def sharedLexikon(self, model):
        """
        Use the Lexikons of another model by creating a reference.
        @param model: object. An instance of a model.
        @return: nothing
        """
        if not isinstance(model, AlignmentModelBase):
            raise RuntimeError("Invalid object, object must be an instance " +
                               "of a sub class of " +
                               "models.modelBase.AlignmentModelBase")
        self.fLex, self.eLex, self.fIndex, self.eIndex =\
            model.fLex, model.eLex, model.fIndex, model.eIndex

    def extendNumpyArray(self, array, shape):
        """
        This method extends an array to a shape not smaller in any dimension
        than target shape. The entries created during extension will all have
        zero values.
        @param array: np.array. The original array.
        @param shape: tuple. The target shape.
        @return: np.array. The extended array.
        """
        if not isinstance(array, np.ndarray):
            array = np.zeros(shape)
            return array
        if len(array.shape) != len(shape):
            raise RuntimeError("Array dimensions doesn't match")
        for i in range(len(shape)):
            if array.shape[i] < shape[i]:
                tmp =\
                    array.shape[0:i] +\
                    (shape[i] - array.shape[i], ) +\
                    array.shape[i + 1:]
                array = np.append(array, np.zeros(tmp), axis=i)
        return array


class TestModelBase(unittest.TestCase):
    def testExtendNumpyArray(self):
        model = AlignmentModelBase()
        arrayA = np.array(range(20)).reshape((4, 5))
        arrayAExtended = np.append(
            np.append(
                arrayA,
                np.zeros((2, 5)),
                axis=0
            ),
            np.zeros((6, 3)),
            axis=1
        )
        np.testing.assert_array_equal(
            arrayAExtended,
            model.extendNumpyArray(arrayA, (6, 8)))
        return


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
