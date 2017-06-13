import os
import sys
import inspect
import optparse
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from fileIO import loadBitext, loadTritext, exportToFile, loadAlignment
from loggers import logging, init_logger
if __name__ == '__main__':
    init_logger('evaluator.log')
logger = logging.getLogger('EVALUATOR')
__version__ = "0.1a"


def evaluate(bitext, result, reference):
    totalAlign = 0
    totalCertain = 0

    totalCertainAlignment = 0
    totalProbableAlignment = 0

    for i in range(min(len(result), len(reference))):
        size_f = len(bitext[i][0])
        size_e = len(bitext[i][1])

        testAlign = []
        for entry in result[i]:
            f = int(entry[0])
            e = int(entry[1])
            if (f > size_f or e > size_e):
                logger.error("NOT A VALID LINK")
                logger.info(i + " " +
                            f + " " + size_f + " " +
                            e + " " + size_e)
            testAlign.append((f, e))

        certainAlign = []
        for entry in reference[i]["certain"]:
            certainAlign.append((entry[0], entry[1]))

        probableAlign = []
        for entry in reference[i]["probable"]:
            probableAlign.append((entry[0], entry[1]))

        # grade
        totalAlign += len(testAlign)
        totalCertain += len(certainAlign)

        totalCertainAlignment += len(
            [item for item in testAlign if item in certainAlign])
        totalProbableAlignment += len(
            [item for item in testAlign if item in certainAlign])
        totalProbableAlignment += len(
            [item for item in testAlign if item in probableAlign])

    precision = float(totalProbableAlignment) / totalAlign
    recall = float(totalCertainAlignment) / totalCertain
    aer = 1 -\
        ((float(totalCertainAlignment + totalProbableAlignment) /
         (totalAlign + totalCertain)))
    fScore = 2 * precision * recall / (precision + recall)

    logger.info("Precision = " + str(precision))
    logger.info("Recall    = " + str(recall))
    logger.info("AER       = " + str(aer))
    logger.info("F-score   = " + str(fScore))
    return {
        "Precision": precision,
        "Recall": recall,
        "AER": aer,
        "F-score": fScore
    }


if __name__ == '__main__':
    # Parsing the options
    optparser = optparse.OptionParser()
    optparser.add_option("--source", dest="source",
                         help="location of source file")
    optparser.add_option("--target", dest="target",
                         help="location of target file")
    optparser.add_option("-v", "--testSize", dest="testSize", default=1956,
                         type="int",
                         help="Number of sentences to use for testing")
    optparser.add_option("-r", "--reference", dest="reference", default="",
                         help="Location of reference file")
    optparser.add_option("-a", "--alignment", dest="alignment", default="",
                         help="Location of alignment file")
    (opts, _) = optparser.parse_args()

    if not opts.source:
        logger.error("source file missing")
    if not opts.target:
        logger.error("target file missing")
    if not opts.reference:
        logger.error("reference file missing")
    if not opts.alignment:
        logger.error("alignment file missing")

    bitext = loadBitext(opts.source, opts.target, opts.testSize)
    alignment = loadAlignment(opts.alignment, opts.testSize)
    goldAlignment = loadAlignment(opts.reference, opts.testSize)

    testAlignment = [sentence["certain"] for sentence in alignment]

    evaluate(bitext, testAlignment, goldAlignment)
