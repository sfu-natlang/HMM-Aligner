# -*- coding: utf-8 -*-

#
# Model Checker of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the model checker that checks if the model implementation fits the
# requirement. It contains information about the APIs of the models.
# If you plan to write your own model, upon loading the model
# checkAlignmentModel will be called to check if the selected model contains
# necessary methods with required parameters.
#
import os
import sys
import inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from models.modelBase import AlignmentModelBase as Base
__version__ = "0.4a"

supportedModels = [
    "IBM1", "IBM1WithAlignmentType", "HMM", "HMMWithAlignmentType"
]

requiredMethods = {
    "train": {"self": "instance",
              "dataset": "list",
              "iterations": "int"},

    "decode": {"self": "instance",
               "dataset": "list"},

    "decodeSentence": {"self": "instance",
                       "sentence": "list"}
}


def checkAlignmentModel(modelClass, logger=True):
    '''
    This function will examine the model class.
    If the return value is False, then the model is unrecognisable by this
    function.
    @param modelClass: class, a model class
    @return: bool. False for unrecognisable
    '''
    if logger:
        from loggers import logging, init_logger
        error = logging.getLogger('CheckModel').error
    else:
        def error(msg):
            print "[ERROR]:", msg

    if not inspect.isclass(modelClass):
        error(
            "Specified Model needs to be a class named AlignmentModel under " +
            "models/ModelName.py")
        return False

    try:
        from models.cModelBase import AlignmentModelBase as cBase
        if issubclass(modelClass, cBase):
            logging.getLogger('CheckModel').info(
                "Loading Cython model, unable to check further")
            return True
    except ImportError:
        pass

    if not issubclass(modelClass, Base):
        error(
            "Specified Model needs to be a subclass of " +
            "models.modelBase.AlignmentModelBase ")
        return False

    try:
        model = modelClass()
    except all:
        error(
            "Specified Model instance cannot be created by calling " +
            "AlignmentModel()")
        return False

    for methodName in requiredMethods:
        method = getattr(modelClass, methodName, None)
        if not callable(method):
            error(
                "Specified Model class needs to have a method called '" +
                methodName + "', " +
                "containing at least the following arguments(without Tag): " +
                str(requiredMethods1[methodName]) + " or the following" +
                "(with Tag) " + str(requiredMethods2[methodName]))
            return False

        args, _, _, _ = inspect.getargspec(method)
        if [a for a in requiredMethods[methodName] if a not in args]:
            error(
                "Specified Model class's '" + methodName + "' method should " +
                "contain the following arguments(with exact same names): " +
                str(requiredMethods[methodName]))
            return False

        if [a for a in args if a not in requiredMethods[methodName]]:
            error(
                "Specified Model class's '" + methodName + "' method should " +
                "contain only the following arguments: " +
                str(requiredMethods1[methodName]))
            return False

        if not [a for a in requiredMethods[methodName] if a not in args]:
            return True
        else:
            error("Unrecognisable model type.")
            return False

    return mode


if __name__ == '__main__':
    print "Launching unit test on: models.modelChecker.checkAlignmentModel"
    print "This test will test the behaviour of checkAlignmentModel on all",\
        "supported models:", supportedModels

    import importlib
    for name in supportedModels:
        try:
            Model = importlib.import_module("models." + name).AlignmentModel
        except all:
            print "Model", name, ": failed"
        mode = checkAlignmentModel(Model, False)
        if mode:
            print "Model", name, ": passed"
        else:
            print "Model", name, ": failed"
