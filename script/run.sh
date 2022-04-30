#!/bin/bash

dataset=test

pngdir=../data/input

outdir=../out/$dataset
tooldir=../tool

#main
python $tooldir/main.py --outdir $outdir --pngdir $pngdir
