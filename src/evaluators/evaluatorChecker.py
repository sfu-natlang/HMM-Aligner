# This is the model checker that checks if the model implementation fits the
# requirement.
import os
import sys
import inspect
import unittest
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from fileIO import loadBitext, loadTritext, exportToFile, loadAlignment

supportedEvaluators = [
    "evaluator", "evaluatorWithType"
]

requiredArguments = {"bitext": "list",
                     "result": "list",
                     "reference": "int"}


def checkEvaluator(func, logger=True):
    def error(msg):
        print "[ERROR]:", msg

    if not inspect.isfunction(func):
        error(
            "Specified Evaluator needs to be a function named evaluate " +
            "under evaluators/EvaluatorName.py")
        return False

    args, _, _, _ = inspect.getargspec(func)
    if [arg for arg in requiredArguments if arg not in args]:
        error(
            "Specified Evaluator function should " +
            "contain the following arguments(with exact same names): " +
            str(requiredArguments))
        return False
    if [arg for arg in args if arg not in requiredArguments]:
        error(
            "Specified Evaluator function should " +
            "contain only the following arguments: " +
            str(requiredArguments))
        return False
    return True


class TestEvaluators(unittest.TestCase):

    def testEvaluatorRegular(self):
        bitext = loadBitext("../support/ut_source.txt",
                            "../support/ut_target.txt")
        noProb = loadAlignment("../support/ut_align_no_prob.a")
        noType = loadAlignment("../support/ut_align_no_type.a")
        certainAlign = [sentence["certain"] for sentence in noProb]
        from evaluator import evaluate
        correctAnswer = {
            "Precision": 1.0,
            "Recall": 1.0,
            "AER": 0.0,
            "F-score": 1.0
        }
        self.assertEqual(evaluate(bitext, certainAlign, noType), correctAnswer)
        return

    def testEvaluatorWithType(self):
        bitext = loadBitext("../support/ut_align_no_tag.cn",
                            "../support/ut_align_no_tag.en")
        original = loadAlignment("../support/ut_align_no_tag.a")
        clean = loadAlignment("../support/ut_align_no_tag_clean.a")
        cleanAll = \
            [sentence["certain"] + sentence["probable"] for sentence in clean]
        from evaluatorWithType import evaluate
        correctAnswer = {
            "Precision": 1.0,
            "Recall": 1.0,
            "AER": 0.0,
            "F-score": 1.0
        }
        self.assertEqual(evaluate(bitext, cleanAll, original), correctAnswer)
        return

    def testEvaluatorWithTypePlusTag(self):
        raise NotImplementedError
        return


if __name__ == '__main__':
    print "Launching unit test on: evaluators"
    print "This test will first check the interface, then the behaviours on ",\
        "all supported evaluators:", supportedEvaluators

    import importlib
    for name in supportedEvaluators:
        try:
            Function = importlib.import_module("evaluators." + name).evaluate
        except all:
            print "Evaluator", name, ": failed"
        if checkEvaluator(Function, False):
            print "Evaluator", name, ": passed"
        else:
            print "Evaluator", name, ": failed"

    print "Now performing behaviour tests"
    unittest.main()
