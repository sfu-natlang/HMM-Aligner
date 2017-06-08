#!/usr/bin/env python
import optparse
import sys
import os

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="~/Daten/align-data", help="data directory (default=~/Daten/align-data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="test.seg.cln", help="prefix of parallel data files (default=test.seg.cln)")
optparser.add_option("-f", "--french", dest="french", default="cn", help="suffix of French (source language) filename (default=cn)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-a", "--alignments", dest="alignment", default="gold.wa", help="suffix of gold alignments filename (default=gold.wa)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_display", dest="n", default=sys.maxint, type="int", help="number of alignments to display")
optparser.add_option("-i", "--inputfile", dest="inputfile", default=None, help="input alignments file (default=sys.stdin)")
(opts, _) = optparser.parse_args()
f_data = os.path.expanduser("%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french))
e_data = os.path.expanduser("%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english))
a_data = os.path.expanduser("%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.alignment))

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

inp = sys.stdin if opts.inputfile is None else file(opts.inputfile)

(size_a, size_s, size_a_and_s, size_a_and_p) = (0.0, 0.0, 0.0, 0.0)
for (i, (f, e, g, a)) in enumerate(zip(open(f_data), open(e_data), open(a_data), inp)):
    fwords = f.strip().split()
    ewords = e.strip().split()
    sure = set([tuple(map(int, x.split("-"))) for x in filter(lambda x: x.find("-") > -1, g.strip().split())])
    possible = set([tuple(map(int, x.split("?"))) for x in filter(lambda x: x.find("?") > -1, g.strip().split())])
    alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
    size_a += len(alignment)
    size_s += len(sure)
    size_a_and_s += len(alignment & sure)
    size_a_and_p += len(alignment & possible) + len(alignment & sure)
    if (i < opts.n):
        sys.stdout.write("    Alignment %i    KEY: ( ) = guessed, * = sure, ? = possible\n" % i)
        sys.stdout.write("    ")
        for j in ewords:
            sys.stdout.write("---")
        sys.stdout.write("\n")
        for (i, f_i) in enumerate(fwords):
            sys.stdout.write(" |")
            for (j, _) in enumerate(ewords):
                (left, right) = ("(", ")") if (i, j) in alignment else (" ", " ")
                point = "*" if (i, j) in sure else "?" if (i, j) in possible else " "
                sys.stdout.write("%s%s%s" % (left, point, right))
            sys.stdout.write(" | %s\n" % f_i)
        sys.stdout.write("    ")
        for j in ewords:
            sys.stdout.write("---")
        sys.stdout.write("\n")
        for k in range(max(map(len, ewords))):
            sys.stdout.write("    ")
            for word in ewords:
                letter = word[k] if len(word) > k else " "
                sys.stdout.write(" %s " % letter)
            sys.stdout.write("\n")
        sys.stdout.write("\n")

precision = size_a_and_p / size_a
recall = size_a_and_s / size_s
aer = 1 - ((size_a_and_s + size_a_and_p) / (size_a + size_s))
sys.stdout.write("Precision = %f\nRecall = %f\nAER = %f\n" % (precision, recall, aer))

for _ in inp:  # avoid pipe error
    pass
