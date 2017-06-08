# The classes here, IntPair and Pair can be replaced by tuples. Such
# replacement will take place at a later stage of development
import time
from copy import deepcopy
from collections import defaultdict
from loggers import logging
from data.Pair import Pair, IntPair


class dummyTask():
    def __init__(self, taskName="Untitled", serial="XXXX"):
        return

    def progress(self, msg):
        return


try:
    from progress import Task
except all:
    Task = dummyTask


class AlignmentModel():
    def __init__(self):
        '''
        @var self.f_count: integer defaultdict with string as index
        @var self.e_count: integer defaultdict with string as index
        @var self.fe_count: integer defaultdict with (str, str) as index
        @var self.tagMap: integer defaultdict with string as index
        @var self.total_f_e_h: float defaultdict with (str, str, int) as index
        '''
        self.t = defaultdict(float)
        self.logger = logging.getLogger('IBM1')
        self.f_count = defaultdict(int)
        self.e_count = defaultdict(int)
        self.fe_count = defaultdict(int)

        self.tagMap = defaultdict(int)
        self.total_f_e_h = defaultdict(float)
        return

    def initialiseTagMap(self):
        self.tagMap["SEM"] = 1
        self.tagMap["FUN"] = 2
        self.tagMap["PDE"] = 3
        self.tagMap["CDE"] = 4
        self.tagMap["MDE"] = 5
        self.tagMap["GIS"] = 6
        self.tagMap["GIF"] = 7
        self.tagMap["COI"] = 8
        self.tagMap["TIN"] = 9
        self.tagMap["NTR"] = 10
        self.tagMap["MTA"] = 11
        return

    def initialiseCountsWithoutSets(self, bitext):
        '''
        Initialises source count, target count and source-target count tables
        (maps)
        @param bitext: bitext of source-target
        '''
        self.initialiseTagMap()

        for (f, e) in bitext:
            # Initialise f_count
            for f_i in f:
                self.f_count[f_i] += 1
                # Initialise fe_count
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1

            # Initialise e_count
            for e_j in e:
                self.e_count[e_j] += 1
        return

    def initialiseCounts(self, tritext, testSize):
        '''
        This method computes source and target counts as well as (source,
        target, alignment type) counts
        (f,e,h) counts are stored in total_f_e_h
        HMMWithAlignmentType initializes its s parameter from total_f_e_h

        @param tritext: string[][]
        @param testSize: int
        '''
        def strip(e_word):
            '''
            @param Word: string
            @return: list of strings
            '''
            indices = ""
            for i in range(len(e_word)):
                e_i = e_word[i]
                if ('0' <= e_i <= '9' or e_i == ','):
                    indices += e_i
            return indices.split(",")

        initialiseTagMap(self)
        sentenceNumber = 1
        for (f, e, wa) in tritext:
            # Initialise f_count
            for f_i in f:
                self.f_count[f_i] += 1
                # Initialise fe_count
                for e_j in e:
                    self.fe_count[(f_i, e_j)] += 1

            # Initialise e_count
            for e_j in e:
                self.e_count[e_j] += 1

            # setting total_f_e_h count
            if (sentenceNumber > len(tritext)):
                for alm in wa:
                    left, right = alm.split("-")
                    leftPositions = strip(left)
                    if (len(leftPositions) == 1 and leftPositions[0] != ""):

                        fWordPos = int(leftPositions[0])
                        fWord = f[fWordPos - 1]

                        rightLength = right.length()

                        linkLabel = right[len(right) - 4: len(right) - 1]
                        engIndices = strip(right[0:len(right) - 5])

                        if (engIndices[0] != ""):
                            for wordIndex in engIndices:
                                engWordPos = int(wordIndex)
                                engWord = E[engWordPos - 1]
                                tagId = tagMap[linkLabel]

                                total_f_e_h[(fWord, engWord, tagId)] += 1

            sentenceNumber += 1

        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        return 1.0 / v

    def gradeAlignmentWithType(self, alignementTestSize, bitext, reference,
                               systemAlignment):
        '''
        @param alignementTestSize: int, number of lines to do test on
        @param bitext: String[][], test bitext
        @param reference: ArrayList<String>, gold reference alignment.
            Including 1-to-many alignments.
        @param systemAlignment: ArrayList<String>, my output alignment
        @return: float, F1 score
        '''
        size_a = 0
        size_s = 0
        size_a_and_s = 0
        size_a_and_p = 0

        for i in range(min(alignementTestSize, len(reference))):
            # alignment of sentence i
            a = systemAlignment[i]
            g = reference.get[i]

            size_f = len(bitext[i][0].strip().split(" "))
            size_e = len(bitext[i][1].strip().split(" "))

            alignment = []
            pairsWithDash = a.strip().split(" ")

            for pwd in pairsWithDash:
                index = pwd.find('-')
                if (index != -1):
                    alignment.append((int(pwd[0, index]), int(pwd[index+1:])))

            for f, e in alignment:
                if (f > size_f or e > size_e):
                    self.logger.error("NOT A VALID LINK")
                    self.logger.info(i + " " +
                                     f + " " + size_f + " " +
                                     e + " " + size_e)

            # grade
            sure = []
            possible = []
            surePairsWithDash = g.strip().split(" ")
            for spwd in surePairsWithDash:
                index = spwd.find('-')
                if (index != -1):
                    engPositions = spwd[index + 1].split(",")
                    for engPos in engPositions:
                        sure.append((int(spwd[0:index]), int(engPos)))

                index = spwd.find('?')
                if (index != -1):
                    possible.append((int(spwd[0:index]), int(spwd[index+1:])))

            size_a += len(alignment)
            size_s += len(sure)
            aAnds = [item for item in alignment if item in sure]
            size_a_and_s += len(aAnds)

            aAndp = [item for item in alignment if item in possible]
            size_a_and_p += len(aAndp) + len(aAnds)

        precision = float(size_a_and_p) / size_a
        recall = float(size_a_and_s) / size_s
        aer = 1 - (float(size_a_and_s + size_a_and_p) / (size_a + size_s))
        fScore = 2 * precision*recall / (precision + recall)
        logger.info("Precision = " + str(precision))
        logger.info("Recall    = " + str(recall))
        logger.info("AER       = " + str(aer))
        logger.info("F-score   = " + str(fScore))
        return fScore

    def gradeAlignmentWithTypeWAPlusTag(self, alignementTestSize, bitext,
                                        reference, systemAlignment):
        '''
        @param alignementTestSize: int, number of lines to do test on
        @param bitext: String[][], test bitext
        @param reference: ArrayList<String>, gold reference alignment with
            alignment types
        @param systemAlignment: ArrayList<String>, my output alignment
        @return: float, F1 score
        '''
        size_a = 0
        size_s = 0
        size_a_and_s = 0
        size_a_and_p = 0

        for i in range(min(alignementTestSize, len(reference))):
            # alignment of sentence i
            a = systemAlignment[i]
            g = reference.get[i]

            size_f = len(bitext[i][0].strip().split(" "))
            size_e = len(bitext[i][1].strip().split(" "))

            alignment = []
            pairsWithDash = a.strip().split(" ")

            for pwd in pairsWithDash:
                index = pwd.find('-')

                if (index != -1):
                    right = pwd[index+1:]
                    rightLength = len[right]
                    engPos = int(right[:rightLength-5])
                    linkLabel = right[rightLength-4: rightLength-1]
                    alignment.append((int(pwd[0:index]), engPos, linkLabel))

            for f, e, tag in alignment:
                if (f > size_f or e > size_e):
                    logger.error("NOT A VALID LINK")
                    logger.info(i + " " +
                                f + " " + size_f + " " +
                                e + " " + size_e)

            # grade
            sure = []

            surePairsWithDash = g.strip().split(" ")
            for spwd in surePairsWithDash:

                index = spwd.find('-')

                if (index != -1):
                    left = spwd[0:index]
                    leftPositions = strip(left)
                    if (len(leftPositions) == 1 and leftPositions[0] != ""):
                        right = spwd[index+1:]
                        rightLength = len(right)
                        linkLabel = right[rightLength-4:rightLength-1]
                        engPositions = strip(right)

                        for engPos in engPositions:
                            if (engPos != ""):
                                sure.append(
                                    (int(leftPositions[0]),
                                     int(engPos),
                                     linkLabel)
                                )

            size_a += len(alignment)
            size_s += len(sure)
            aAnds = [item for item in alignment if item in sure]
            size_a_and_s += len(aAnds)

            size_a_and_p += len(aAnds)

        precision = float(size_a_and_p) / size_a
        recall = float(size_a_and_s) / size_s
        aer = 1 - (float(size_a_and_s + size_a_and_p) / (size_a + size_s))
        fScore = 2 * precision * recall / (precision + recall)
        logger.info("Precision = " + str(precision))
        logger.info("Recall    = " + str(recall))
        logger.info("AER       = " + str(aer))
        logger.info("F-score   = " + str(fScore))
        return fScore

    def gradeAlign(self, alignementTestSize, bitext, reference,
                   systemAlignment):
        '''
        @param alignementTestSize: int, number of lines to do test on
        @param bitext: String[][], test bitext
        @param reference: ArrayList<String>, gold reference with sure and
            possible links
        @param systemAlignment: ArrayList<String>, my output alignment
        @return NaN
        '''
        size_a = 0
        size_s = 0
        size_a_and_s = 0
        size_a_and_p = 0

        for i in range(min(alignementTestSize, len(reference))):
            # alignment of sentence i
            a = systemAlignment[i]
            g = reference[i]

            size_f = len(bitext[i][0].strip().split(" "))
            size_e = len(bitext[i][1].strip().split(" "))

            alignment = []
            pairsWithDash = a.strip().split(" ")

            for pwd in pairsWithDash:
                index = pwd.find('-')
                if (index != -1):
                    alignment.append((int(pwd[0:index]), int(pwd[index+1:])))

            for f, e in alignment:
                if (f > size_f or e > size_e):
                    logger.error("NOT A VALID LINK")
                    logger.info(i + " " +
                                f + " " + size_f + " " +
                                e + " " + size_e)

            # grade
            sure = []
            possible = []
            surePairsWithDash = g.strip().split(" ")
            for spwd in surePairsWithDash:
                index = spwd.find('-')
                if (index != -1):
                    sure.append((int(spwd[0:index]), int(spwd[index+1:])))
                index = spwd.find('?')
                if (index != -1):
                    possible.append((int(spwd[0:index]), int(spwd[index+1:])))

            size_a += len(alignment)
            size_s += len(sure)
            aAnds = [item for item in alignment if item in sure]
            size_a_and_s += len(aAnds)

            aAndp = [item for item in alignment if item in possible]
            size_a_and_p += len(aAndp) + len(aAnds)

        precision = float(size_a_and_p) / size_a
        recall = float(size_a_and_s) / size_s
        aer = 1 - (float(size_a_and_s + size_a_and_p) / (size_a + size_s))
        logger.info("Precision = " + str(precision))
        logger.info("Recall    = " + str(recall))
        logger.info("AER       = " + str(aer))
        return

    def convertFileToArrayList(self, fileName):
        '''
        Converts a file to an ArrayList of String (line by line)
        @param fileName: String
        @return: ArrayList<String>, an ArrayList of String where each element
            of the list is a line of fileName
        '''
        return [sentence.strip().split() for sentence in open(fileName)]

    def train(self, bitext, iterations=5):
        task = Task("Aligner", "IBM1NI" + str(iterations))
        self.logger.info("Starting Training Process")
        self.logger.info("Training size: " + str(len(bitext)))
        start_time = time.time()

        self.initialiseCountsWithoutSets(bitext)
        initialValue = 1.0 / len(self.f_count)
        for key in self.fe_count:
            self.t[key] = initialValue
        self.logger.info("Initialisation Complete")

        for iteration in range(iterations):
            c = defaultdict(float)
            total = defaultdict(float)
            self.logger.info("Starting Iteration " + str(iteration))
            counter = 0

            for (f, e) in bitext:
                counter += 1
                task.progress("IBM1New iter %d, %d of %d" %
                              (iteration, counter, len(bitext),))
                for fWord in f:
                    z = 0
                    for eWord in e:
                        z += self.t[(fWord, eWord)]
                    for eWord in e:
                        c[(fWord, eWord)] += self.t[(fWord, eWord)] / z
                        total[eWord] += self.t[(fWord, eWord)] / z

            for (f, e) in self.fe_count:
                self.t[(f, e)] = c[(f, e)] / total[e]

        end_time = time.time()
        self.logger.info("Training Complete, total time(seconds): %f" %
                         (end_time - start_time,))
        return

    def decode(self, bitext):
        self.logger.info("Start decoding")
        self.logger.info("Testing size: " + str(len(bitext)))
        result = []

        for (f, e) in bitext:
            sentenceAlignment = []

            for i in range(len(f)):
                max_t = 0
                argmax = -1
                for j in range(len(e)):
                    t = self.tProbability(f[i], e[j])
                    if t > max_t:
                        max_t = t
                        argmax = j
                sentenceAlignment.append((i, argmax))

            result.append(sentenceAlignment)
        self.logger.info("Decoding Complete")
        return result
