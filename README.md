# HMM-Aligner
This is the implementation of word aligner using Hidden Markov Model.

## Note
The HMM Aligner is still under development, so new features and optimisations
will be added. However we always maintain a master branch that works. To see
how it works please checkout our
[Wiki](https://github.com/sfu-natlang/HMM-Aligner/wiki) page. For development
progress please checkout the
[Project](https://github.com/sfu-natlang/HMM-Aligner/projects) page.

Currently the master branch uses Numpy to speed up the training. They are
however unfortunately a bit harder to read of course, so the old versions are
kept in `src/models/old`. If you copy these models out they will be able to run
as they are using the same API as the Numpy versions, there are minor
differences in the decoding part but the training code are essentially the
same, if you want to understand how our aligner works they should be very
useful.

## Get Started

To use the models starting with `c`, for example `cIBM1`, one must compile the
files needed.

    > cd src
    > python setup.py build_ext --inplace

The other models directly under the directory `src/models` can be directly
used.

## Usage

    > cd src
    > python aligner.py -h

For detailed specifications, please checkout our
[Wiki](https://github.com/sfu-natlang/HMM-Aligner/wiki) page for API specs.
