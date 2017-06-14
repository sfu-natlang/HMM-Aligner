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
__version__ = "0.2a"

supportedModels = [
    "IBM1Old", "IBM1New", "HMMOld",
    "IBM1Type"
]

requiredMethods1 = {
    "train": {"self": "instance",
              "bitext": "list",
              "iterations": "int"},

    "decode": {"self": "instance",
               "bitext": "list"}
}

requiredMethods2 = {
    "train": {"self": "instance",
              "formTritext": "list",
              "tagTritext": "list",
              "iterations": "int"},

    "decode": {"self": "instance",
               "formBitext": "list",
               "tagBitext": "list"}
}


def checkAlignmentModel(modelClass, logger=True):
    '''
    There are two types of models supported, type 1 trains on bitext and
    type 2 on a tritext of the original text and a tritext of tags(POS).
    This function will examine the model class and determine its type.
    If the return value is -1, then the model is unrecognisable by this
    function.
    @param modelClass: class, a model class
    @return: int, model type. -1 for unrecognisable
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
        return -1

    mode = -1

    for methodName in requiredMethods1:
        method = getattr(modelClass, methodName, None)
        if not callable(method):
            error(
                "Specified Model class needs to have a method called '" +
                methodName + "', " +
                "containing at least the following arguments(without Tag): " +
                str(requiredMethods1[methodName]) + " or the following" +
                "(with Tag) " + str(requiredMethods2[methodName]))
            return -1

        args, _, _, _ = inspect.getargspec(method)
        if ([a for a in requiredMethods1[methodName] if a not in args] and
                [a for a in requiredMethods2[methodName] if a not in args]):
            error(
                "Specified Model class's '" + methodName + "' method should " +
                "contain the following arguments(with exact same names): " +
                str(requiredMethods1[methodName]) + " or the following" +
                "(with Tag) " + str(requiredMethods2[methodName]))
            return -1

        if ([a for a in args if a not in requiredMethods1[methodName]] and
                [a for a in args if a not in requiredMethods2[methodName]]):
            error(
                "Specified Model class's '" + methodName + "' method should " +
                "contain only the following arguments: " +
                str(requiredMethods1[methodName]) + " or the following" +
                "(with Tag) " + str(requiredMethods2[methodName]))
            return -1

        if not([a for a in requiredMethods1[methodName] if a not in args] +
                [a for a in args if a not in requiredMethods1[methodName]]):
            if mode == -1 or mode == 1:
                mode = 1
            else:
                error("Unrecognisable model type.")

        if not([a for a in requiredMethods2[methodName] if a not in args] +
                [a for a in args if a not in requiredMethods2[methodName]]):
            if mode == -1 or mode == 2:
                mode = 2
            else:
                error("Unrecognisable model type.")

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
        if mode != -1:
            print "Model", name, ": passed, type", mode
        else:
            print "Model", name, ": failed"
