import os
import sys


def exportToFile(result, fileName):
    outputFile = open(fileName, "w")
    for sentenceAlignment in result:
        line = ""
        for (i, j) in result:
            line += str(i) + "-" + str(j) + " "

        outputFile.write(line + "\n")
    outputFile.close()
    return


def loadBitext(file1, file2, linesToLoad):
    path1 = os.path.expanduser(file1)
    path2 = os.path.expanduser(file2)
    bitext =\
        [[sentence.strip().split() for sentence in pair] for pair in
            zip(open(path1), open(path2))[:linesToLoad]]
    return bitext


def loadTritext(file1, file2, file3, linesToLoad):
    path1 = os.path.expanduser(file1)
    path2 = os.path.expanduser(file2)
    path3 = os.path.expanduser(file3)
    tritext =\
        [[sentence.strip().split() for sentence in trio] for trio in
            zip(open(path1), open(path2), open(path3))[:linesToLoad]]
    return tritext


def loadAlignment(fileName, linesToLoad=sys.maxint):
    content =\
        [sentence.strip().split() for sentence in open(fileName)[:linesToLoad]]

    result = []

    for sentence in content:
        certainAlign = []
        probableAlign = []
        for entry in sentence.strip().split():
            # Every entry is expected to be of the format: "NN-NN,NN" or
            # "NN?NN,NN", where NNs are numbers
            if entry.find('-') != -1:
                items = entry.split('-')
                f = int(item[0])
                for e in item[1].split(','):
                    if len(item) > 2:
                        certainAlign.append((f, e, item[2]))
                    else:
                        certainAlign.append((f, e))

            elif entry.find('?') != -1:
                items = entry.split('?')
                f = int(item[0])
                for e in item[1].split(','):
                    if len(item) > 2:
                        probableAlign.append((f, e, item[2]))
                    else:
                        probableAlign.append((f, e))

        sentenceAlignment = {"certain": certainAlign,
                             "probable": probableAlign}
        result.append(sentenceAlignment)

    return result
