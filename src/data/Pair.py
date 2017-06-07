# The classes here, IntPair and Pair can be replaced by tuples. Such
# replacement will take place at a later stage of development


def stringHashcode(s):
    if not s:
        return 0
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000


class IntPair():
    def __init__(self, left=0, right=0):
        '''
        left: int
        right: int
        '''
        self.left = left
        self.right = right
        return

    def __eq__(self, other):
        if type(other) is not IntPair:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        return False

    def hashCode(self):
        '''
        @return int
        '''
        prime = 31
        return (1 * prime + self.left) * prime + self.right

    def toString(self):
        '''
        @return string
        '''
        return "(" + str(self.left) + ", " + str(self.right) + ")"


class Pair():
    _serialVErsionUID = 1L

    def __init__(self, left="", right=""):
        '''
        @param left: string
        @param right: string
        @return NaN
        '''
        self.left = left
        self.right = right
        return

    def setPair(self, left, right):
        '''
        @param left: string
        @param right: string
        @return NaN
        '''
        self.left = left
        self.right = right
        return

    def __eq__(self, other):
        if type(other) is not IntPair:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        return False

    def hashCode(self):
        '''
        @return int
        '''
        prime = 31
        result = 1
        result = prime * result + stringHashcode(self.left)
        result = prime * result + stringHashcode(self.right)
        return result

    def toString(self):
        '''
        @return string
        '''
        return "(" + str(self.left) + ", " + str(self.right) + ")"
