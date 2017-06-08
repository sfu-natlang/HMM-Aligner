# This is the model checker that checks if the model implementation fits the
# requirement.
import os
import sys
import inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

supportedModels = [
    "IBM1Old", "IBM1New"
]

requiredMethods = {
    "train": {"bitext": list,
              "iterations": int},
    "decode": {"bitext": list}
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

    trainMethod = getattr(modelClass, "train", None)
    if not callable(trainMethod):
        error(
            "Specified Model class needs to have a method called train, " +
            "containing at least the following arguments: " +
            "bitext(list of (str, str)), iterations(int)")
        return False

    decodeMethod = getattr(modelClass, "decode", None)
    if not callable(decodeMethod):
        error(
            "Specified Model class needs to have a method called " +
            "decode, containing at least the following " +
            "arguments: bitext(list of (str, str)), iterations(int)")
        return False
    return True


if __name__ == '__main__':
    print "Launching unit test on: models.modelChecker.checkAlignmentModel"
    print "This test will test the behaviour of checkAlignmentModel on all",\
        "supported models:\n", supportedModels

    import importlib
    for name in supportedModels:
        Model = importlib.import_module("models." + name).AlignmentModel
        if checkAlignmentModel(Model, False):
            print "Model", name, ": passed"
        else:
            print "Model", name, ": failed"
