# The classes here, IntPair and Pair can be replaced by tuples. Such
# replacement will take place at a later stage of development


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
        if not self.left:
            result = prime * result + 0
        else:
            result = prime * result + self.left.hashCode()
        if not self.right:
            result = prime * result + 0
        else:
            result = prime * result + self.right.hashCode()
        return result

    def toString(self):
        '''
        @return string
        '''
        return "(" + str(self.left) + ", " + str(self.right) + ")"
