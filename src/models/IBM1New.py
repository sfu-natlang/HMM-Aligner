# The classes here, IntPair and Pair can be replaced by tuples. Such
# replacement will take place at a later stage of development
from data.Pair import Pair, IntPair


class IBM1_model():
    def __init__(self):
        raise NotImplementedError
        return

    def initialiseTagMap(self):
        raise NotImplementedError
        return

    def readBitext(self, fileName, fileName2, linesToRead):
        '''
        @param fileName: String, source file name
        @param fileName2: String, target file name
        @param linesToRead: int, number of lines to read from both files to
            create the bitext
        @return string array, [linesToRead][2]
        '''
        raise NotImplementedError
        return None

    def readTritext(self, fileName, fileName2, fileName3, linesToRead):
        '''
        @param fileName: String, source file name
        @param fileName2: String, target file name
        @param fileName3: String, gold word alignment and alignment type file
            name
        @param linesToRead: int, number of lines to read from both files to
            create the bitext
        @return string array, [linesToRead][2]
        '''
        raise NotImplementedError
        return

    def initialiseCountsWithoutSets(self, bitext):
        '''
        Initialises source count, target count and source-target count tables
        (maps)
        @param bitext: bitext of source-target
        '''
        self.initialiseTagMap()

        raise NotImplementedError
        return None

    def initialiseCounts(self, tritext, testSize):
        '''
        This method computes source and target counts as well as (source,
        target, alignment type) counts
        (f,e,h) counts are stored in total_f_e_h
        HMMWithAlignmentType initializes its s parameter from total_f_e_h

        @param tritext: string[][]
        @param testSize: int
        '''

        raise NotImplementedError
        return None

    def strip(self, englishWord):
        '''
        @param englishWord: string
        @return: String[]
        '''
        raise NotImplementedError
        return None

    def EM_IBM1(self, sourceCount, targetCount, st_count, bitext):
        '''
        EM for IBM1
        @param sourceCount: HashMap<String, Integer>
        @param targetCount: HashMap<String, Integer>
        @param st_count: HashMap<Pair, Integer>
        @param bitext: String[][]
        @return t parameter: HashMap<Pair, Double>
        '''

        raise NotImplementedError
        return None

    def tProbability(self, f, e, t_table):
        '''
        Smoothes t(f|e) parameter by backing-off to a uniform probability 1/V
        @param f: String, source word
        @param e: String, target word
        @param t_table: HashMap<Pair, Double>
        @return: float, the somoothed t(f|e)
        '''
        raise NotImplementedError
        return None

    def print_alignment_SD_ibm1(self, bitext, t, alignmentFile):
        '''
        @param bitext: String[][]
        @param t: HashMap<Pair, Double>
        @param alignmentFile: String
        @return: ArrayList<String>
        '''
        raise NotImplementedError
        return None

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
        raise NotImplementedError
        return None

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
        raise NotImplementedError
        return None

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
        raise NotImplementedError
        return

    def convertFileToArrayList(self, fileName):
        '''
        Converts a file to an ArrayList of String (line by line)
        @param fileName: String
        @return: ArrayList<String>, an ArrayList of String where each element
            of the list is a line of fileName
        '''
        raise NotImplementedError
        return None

    def mainIBM1(self, trainingSize, testSize, trainPrefix, sourceLang,
                 targetLang, testPrefix, referenceFile, alignmentFile):
        '''
        @param trainingSize: int
        @param testSize: int
        @param trainPrefix: String
        @param sourceLang: String
        @param targetLang: String
        @param testPrefix: String
        @param referenceFile: String
        @param alignmentFile: String
        @return: NaN
        '''
        raise NotImplementedError
        return

AlignmentModel = IBM1_model
