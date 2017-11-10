import optparse
from fileIO import loadAlignment, exportToFile, intersect
__version__ = "0.1a"


if __name__ == "__main__":
    # Parsing the options
    optparser = optparse.OptionParser()
    optparser.add_option("-f", "--forward", dest="forward",
                         help="Forward alignment file")
    optparser.add_option("-r", "--reverse", dest="reverse",
                         help="Reversed alignment file")
    optparser.add_option("-o", "--output", dest="output",
                         help="Location of output file")
    (opts, _) = optparser.parse_args()

    st = loadAlignment(opts.forward)
    ts = loadAlignment(opts.reverse)
    st = [sent["certain"] for sent in st]
    ts = [sent["certain"] for sent in ts]
    result = intersect(st, ts)
    exportToFile(result, opts.output)
