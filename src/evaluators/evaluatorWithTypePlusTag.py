from loggers import logging
logger = logging.getLogger('EVALUATOR')


def evaluate(bitext, result, reference):
    totalAlign = 0
    totalCertain = 0

    totalCertainAlignment = 0
    totalProbableAlignment = 0

    for i in range(min(len(result), len(reference))):
        testAlign = result[i]
        goldAlign = reference[i]

        size_f = len(bitext[i][0])
        size_e = len(bitext[i][1])

        testAlign = []
        for entry in testSentence.strip.split():
            # Every entry is expected to be of the format: "NN-NN", where NN is
            # a number
            if (entry.find('-') != -1):
                fStr, eStr, tag = pwd.split('-')
                f = int(fStr)
                e = int(eStr)
                if (f > size_f or e > size_e):
                    logger.error("NOT A VALID LINK")
                    logger.info(i + " " +
                                f + " " + size_f + " " +
                                e + " " + size_e)
                testAlign.append((f, e, tag))

        # grade
        certainAlign = goldAlign["certain"]
        probableAlign = goldAlign["probable"]

        totalAlign += len(testAlign)
        totalCertain += len(certainAlign)

        totalCertainAlignment += len(
            [item for item in testAlign if item in certainAlign])
        totalProbableAlignment += len(
            [item for item in testAlign if item in certainAlign])

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
    return
