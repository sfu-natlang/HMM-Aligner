# -*- coding: utf-8 -*-

#
# FileIO of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# Here are all of the functions used in HMM aligner to handle file operations.
# Which includes: reading text file, reading gold alignment files and export
# alignment to file. Exported alignment can also be read using the same
# function to read gold alignment files.
#
import os
import sys
import inspect
import unittest
__version__ = "0.3a"


def exportToFile(result, fileName):
    '''
    This function is used to export alignment to file

    @param result: Alignment, detailed description of this format:
        https://github.com/sfu-natlang/HMM-Aligner/wiki/API-reference:-Alignment-Data-Format-V0.1a#alignment
    @param fileName: str, the file to export to
    '''
    outputFile = open(fileName, "w")
    for sentenceAlignment in result:
        line = ""
        for item in sentenceAlignment:
            if len(item) == 2:
                line += str(item[0]) + "-" + str(item[1]) + " "
            if len(item) == 3:
                line += str(item[0]) + "-" + str(item[1]) +\
                    "(" + str(item[2]) + ") "

        outputFile.write(line + "\n")
    outputFile.close()
    return


def _loadBitext(file1, file2, linesToLoad=sys.maxint):
    '''
    This function is used to read a bitext from two text files.

    @param file1: str, the first file to read
    @param file2: str, the second file to read
    @param* linesToLoad: int, the lines to read
    @return: Bitext, detailed of this format:
        https://github.com/sfu-natlang/HMM-Aligner/wiki/API-reference:-Dataset-Data-Format-V0.1a#bitext
    '''
    path1 = os.path.expanduser(file1)
    path2 = os.path.expanduser(file2)
    bitext =\
        [[sentence.strip().split() for sentence in pair] for pair in
            zip(open(path1), open(path2))[:linesToLoad]]
    return bitext


def _loadTritext(file1, file2, file3, linesToLoad=sys.maxint):
    '''
    This function is used to read a bitext from two text files.

    @param file1: str, the first file to read
    @param file2: str, the second file to read
    @param file3: str, the third file to read
    @param* linesToLoad: int, the lines to read
    @return: Tritext, detail of this format:
        https://github.com/sfu-natlang/HMM-Aligner/wiki/API-reference:-Dataset-Data-Format-V0.1a#tritext
    '''
    path1 = os.path.expanduser(file1)
    path2 = os.path.expanduser(file2)
    path3 = os.path.expanduser(file3)
    tritext =\
        [[sentence.strip().split() for sentence in trio] for trio in
            zip(open(path1), open(path2), open(path3))[:linesToLoad]]
    return tritext


def processAlignmentEntry(entry, listToAddTo, splitChar='-'):
    if entry.find(splitChar) != -1:
        for ch in (',', '(', ')', '[', ']'):
            entry = entry.replace(ch, splitChar)
        items = entry.split(splitChar)
        f = int(items[0])
        alignmentType = ""
        for i in range(len(items) - 1, 0, -1):
            if items[i].isdigit():
                e = int(items[i])
                if alignmentType != "":
                    listToAddTo.append((f, e, alignmentType))
                else:
                    listToAddTo.append((f, e))
            else:
                alignmentType = items[i]
    return


def loadDataset(fFiles, eFiles, alignmentFile="", linesToLoad=sys.maxint):
    '''
    This function is used to read a Dataset files.

    @param fFiles: list of str, the file containing source language files,
        including FORM, POS, etc.,
    @param eFiles: list of str, the file containing target language files,
        including FORM, POS, etc.,
    @param alignmentFile: str, the alignmentFile
    @param* linesToLoad: int, the lines to read
    @return: Dataset, detail of this format:
        https://github.com/sfu-natlang/HMM-Aligner/wiki/API-reference:-Dataset-Data-Format-V0.2a#tritext
    '''
    fContents =\
        [zip(*[fContent.strip().split() for fContent in contents])
         for contents in zip(*[open(os.path.expanduser(fFile))
                             for fFile in fFiles])[:linesToLoad]]
    eContents =\
        [zip(*[eContent.strip().split() for eContent in contents])
         for contents in zip(*[open(os.path.expanduser(eFile))
                             for eFile in eFiles])[:linesToLoad]]

    if alignmentFile:
        alignment =\
            [sentence[0].strip().split() for sentence in
                zip(open(os.path.expanduser(alignmentFile)))[:linesToLoad]]

        for i in range(len(alignment)):
            entries = alignment[i]
            result = []
            for entry in entries:
                processAlignmentEntry(entry, result)
            alignment[i] = result
    else:
        alignment = [[] for i in range(min(linesToLoad,
                                           len(fContents),
                                           len(eContents)))]
    return zip(fContents, eContents, alignment)


def loadAlignment(fileName, linesToLoad=sys.maxint):
    '''
    This function is used to read the GoldAlignment or Alignment from files.

    @param fileName: str, the Alignment file to read
    @param* linesToLoad: int, the lines to read
    @return: GoldAlignment, detail of this format:
        https://github.com/sfu-natlang/HMM-Aligner/wiki/API-reference:-Alignment-Data-Format-V0.1a#goldalignment
    '''
    content =\
        [line.strip().split() for line in open(fileName)][:linesToLoad]

    result = []

    for sentence in content:
        certainAlign = []
        probableAlign = []
        for entry in sentence:
            if entry.find('-') != -1:
                processAlignmentEntry(entry, certainAlign, splitChar='-')

            elif entry.find('?') != -1:
                processAlignmentEntry(entry, probableAlign, splitChar='?')

        sentenceAlignment = {"certain": certainAlign,
                             "probable": probableAlign}
        result.append(sentenceAlignment)

    return result


class TestFileIO(unittest.TestCase):

    def testLoadBitext(self):
        bitext1 = _loadBitext("support/ut_source.txt", "support/ut_target.txt")
        bitext2 = _loadBitext("support/ut_target.txt", "support/ut_source.txt")
        for (f1, e1), (e2, f2) in zip(bitext1, bitext2):
            self.assertSequenceEqual(f1, f2)
            self.assertSequenceEqual(e1, e2)
        return

    def testLoadTritext(self):
        tritext1 = _loadTritext("support/ut_source.txt",
                                "support/ut_target.txt",
                                "support/ut_target.txt")
        tritext2 = _loadTritext("support/ut_target.txt",
                                "support/ut_source.txt",
                                "support/ut_target.txt")
        for (f1, e1, t1), (f2, e2, t2) in zip(tritext1, tritext2):
            self.assertSequenceEqual(f1, e2)
            self.assertSequenceEqual(e1, t1)
            self.assertSequenceEqual(e1, f2)
            self.assertSequenceEqual(e1, t2)
        return

    def testExportToFile(self):
        alignment = loadAlignment("support/ut_align_no_prob.a")
        certainAlign = [sentence["certain"] for sentence in alignment]
        exportToFile(certainAlign, "support/ut_align_no_prob.exported.wa")
        compare = _loadBitext("support/ut_align_no_prob.a",
                              "support/ut_align_no_prob.exported.wa")
        for (s1, s2) in compare:
            self.assertItemsEqual(s1, s2)
        return

    def testLoadAlignmentWithoutType(self):
        alignment = loadAlignment("support/ut_align_no_type.a")
        certainAlign = [sentence["certain"] for sentence in alignment]
        probableAlign = [sentence["probable"] for sentence in alignment]
        exportToFile(certainAlign, "support/ut_align_no_type.certain.wa")
        exportToFile(probableAlign, "support/ut_align_no_type.probable.wa")
        loadedCertain = loadAlignment("support/ut_align_no_type.certain.wa")
        loadedProbable = loadAlignment("support/ut_align_no_type.probable.wa")

        for (f1, f2) in zip(loadedCertain, certainAlign):
            self.assertItemsEqual(f1["certain"], f2)

        for (f1, f2) in zip(loadedProbable, probableAlign):
            self.assertItemsEqual(f1["certain"], f2)

        return

    def testLoadAlignmentWithoutTag(self):
        alignment1 = loadAlignment("support/ut_align_no_tag.a")
        loadedCertain = [sentence["certain"] for sentence in alignment1]
        loadedProbable = [sentence["probable"] for sentence in alignment1]

        alignment2 = loadAlignment("support/ut_align_no_tag_clean.a")
        certainAlign = [sentence["certain"] for sentence in alignment2]
        probableAlign = [sentence["probable"] for sentence in alignment2]

        for (f1, f2) in zip(loadedCertain, certainAlign):
            self.assertItemsEqual(f1, f2)

        for (f1, f2) in zip(loadedProbable, probableAlign):
            self.assertItemsEqual(f1, f2)

        return

    def testLoadDataset(self):
        f1 = f2 = "support/ut_source.txt"
        e1 = e2 = "support/ut_target.txt"
        alignFile = "support/ut_align_no_type.a"
        dataset = loadDataset((f1, f2), (e1, e2))
        for (f, e, alignment) in dataset:
            for f1, f2 in f:
                self.assertItemsEqual(f1, f2)
            for e1, e2 in e:
                self.assertItemsEqual(e1, e2)


if __name__ == '__main__':
    unittest.main()
