# HMM-Aligner
This is the implementation of word aligner using Hidden Markov Model.

### How to run (align.py uses HMM model)
Note: Due to the fact that Python is a dynamically interpreted language, and training the HMM model involves a lot of nested loops(line 150~170, align.py), it takes about **7 hours** to train on the whole dataset. Please be patient while waiting.

     python src/align.py -p europarl -f de > output.a

## HMM model

Following the research [http://dl.acm.org/citation.cfm?id=778824](http://dl.acm.org/citation.cfm?id=778824), we trained an HMM model using the F\_count, Fe\_count, and translation table from IBM1 model.

## IBM1 model
Our implementation of the aligner will utilise the IBM model 1 scores produced by our IBM1\_model, which itself would function as an aligner as well. 

The code is in src/method_IBM1.py.

      for each (f, e):
        for each fi in f:
          find ej that maximize t(fi|ej)
          align fi to ej

Note that this implementation itself doesn't produce very satisfying alignments.