# -*- coding: utf-8 -*-

#
# HMM model implementation(old) of HMM Aligner
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
__version__ = "0.1a"

supportedModels = [
    "IBM1Old", "IBM1New", "HMMOld"
]

requiredMethods = {
    "train": {"self": "instance",
              "bitext": "list",
              "iterations": "int"},

    "decode": {"self": "instance",
               "bitext": "list"}
}


def checkAlignmentModel(modelClass, logger=True):
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

    for methodName in requiredMethods:
        method = getattr(modelClass, methodName, None)
        if not callable(method):
            error(
                "Specified Model class needs to have a method called '" +
                methodName + "', " +
                "containing at least the following arguments: " +
                str(requiredMethods[methodName]))
            return False
        args, _, _, _ = inspect.getargspec(method)
        if [arg for arg in requiredMethods[methodName] if arg not in args]:
            error(
                "Specified Model class's '" + methodName + "' method should " +
                "contain the following arguments(with exact same names): " +
                str(requiredMethods[methodName]))
            return False
        if [arg for arg in args if arg not in requiredMethods[methodName]]:
            error(
                "Specified Model class's '" + methodName + "' method should " +
                "contain only the following arguments: " +
                str(requiredMethods[methodName]))
            return False

    return True


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
        if checkAlignmentModel(Model, False):
            print "Model", name, ": passed"
        else:
            print "Model", name, ": failed"
