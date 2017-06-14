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
__version__ = "0.2a"


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


def loadBitext(file1, file2, linesToLoad=sys.maxint):
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


def loadTritext(file1, file2, file3, linesToLoad=sys.maxint):
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
            # Every entry is expected to be of the format: "NN-NN,NN" or
            # "NN?NN,NN", where NNs are numbers
            if entry.find('-') != -1:
                for ch in ('(', ')', '[', ']'):
                    entry = entry.replace(ch, '-')
                items = entry.split('-')
                f = int(items[0])
                for eStr in items[1].split(','):
                    if eStr == '':
                        continue
                    e = int(eStr)
                    if len(items) > 2:
                        certainAlign.append((f, e, items[2]))
                    else:
                        certainAlign.append((f, e))

            elif entry.find('?') != -1:
                for ch in ('(', ')', '[', ']'):
                    entry = entry.replace(ch, '?')
                items = entry.split('?')
                f = int(items[0])
                for eStr in items[1].split(','):
                    e = int(eStr)
                    if len(items) > 2:
                        probableAlign.append((f, e, items[2]))
                    else:
                        probableAlign.append((f, e))

        sentenceAlignment = {"certain": certainAlign,
                             "probable": probableAlign}
        result.append(sentenceAlignment)

    return result


class TestFileIO(unittest.TestCase):

    def testLoadBitext(self):
        bitext1 = loadBitext("support/ut_source.txt", "support/ut_target.txt")
        bitext2 = loadBitext("support/ut_target.txt", "support/ut_source.txt")
        for (f1, e1), (e2, f2) in zip(bitext1, bitext2):
            self.assertSequenceEqual(f1, f2)
            self.assertSequenceEqual(e1, e2)
        return

    def testLoadTritext(self):
        tritext1 = loadTritext("support/ut_source.txt",
                               "support/ut_target.txt",
                               "support/ut_target.txt")
        tritext2 = loadTritext("support/ut_target.txt",
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
        compare = loadBitext("support/ut_align_no_prob.a",
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


if __name__ == '__main__':
    unittest.main()
