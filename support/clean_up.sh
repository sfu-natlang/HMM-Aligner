#!/bin/bash

SRCDIR=$(dirname "$0")/../src
rm $SRCDIR/*.pyc
rm $SRCDIR/*/*.pyc
rm $SRCDIR/*/*.c
rm $SRCDIR/*/*.so
