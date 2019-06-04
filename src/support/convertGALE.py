#
# Converter for GALE datasets
# To be more specific, this below works with Arabic-English datasets with
# POS Tags
#
import sys
import os


def pennTreeIntoTags(fileName, linesToLoad=sys.maxsize):
    result = []
    fileName = os.path.expanduser(fileName)
    content = [line.strip() for line in open(fileName)][:linesToLoad]
    for line in content:
        sentence = []
        procLine = line
        for ch in ('(', ')', '[', ']'):
            procLine = procLine.replace(ch, ' ')
        procLine = procLine.split()
        for i in range(len(procLine)):
            if procLine[i].isdigit():
                sentence.append(procLine[i - 1])
                print("Missing elements!")
        # sentence = [entry for entry in sentence if entry != "-NONE-"]
        result.append(sentence)
    return result


def tokenIntoForms(fileName, linesToLoad=sys.maxsize):
    result = []
    fileName = os.path.expanduser(fileName)
    content = [line.strip() for line in open(fileName)][:linesToLoad]
    for line in content:
        sentence = []
        procLine = line.split()
        for i in range(len(procLine)):
            sentence.append(procLine[i - 1].split(";")[-1])
        # sentence = [entry for entry in sentence if entry != "*"]
        sentence = [entry for entry in sentence]
        result.append(sentence)
    return result


def alignmentToList(fileName, linesToLoad=sys.maxsize):
    result = []
    fileName = os.path.expanduser(fileName)
    content = [line.strip() for line in open(fileName)][:linesToLoad]
    for line in content:
        result.append(line.split())
    return result


def convertFiles(filePattern, converter, output="o.pos"):
    result = []
    import glob
    for name in glob.glob(filePattern):
        if os.path.isfile(name):
            print(name)
            result += converter(name)
    file = open(output, "w")
    for sentence in result:
        file.write(" ".join(sentence))
        file.write("\n")
    file.close()
    return


if __name__ == '__main__':
    convertFiles(
        "tree/ATB/*.tree", pennTreeIntoTags, "train.tag")
    convertFiles(
        "tree/EATB/*.tree", pennTreeIntoTags, "train.tag")
    convertFiles(
        "source/tokenized/*.tkn", tokenIntoForms, "train.form")
    convertFiles(
        "translation/tokenized/*.tkn", tokenIntoForms, "train.form")
    convertFiles(
        "WA/*.wa", tokenIntoForms, "gold.wa")
