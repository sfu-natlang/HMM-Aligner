# -*- coding: utf-8 -*-

#
# IBM model 1 with alignment type implementation of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This is the implementation of IBM model 1 word aligner with alignment type.
#
from collections import defaultdict
from loggers import logging
from models.IBM1Base import AlignmentModelBase as IBM1Base

class AlignmentModel():
    def __init__(self):
        self.logger = logging.getLogger('IBM1')
        return
        
