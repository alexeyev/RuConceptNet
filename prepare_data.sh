#!/bin/bash
echo
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
mkdir data
grep $'/ru/\S*\t\S*/ru/' conceptnet-assertions-5.7.0.csv > data/russian-conceptnet.tsv
python conceptnet/compressor.py
# rm conceptnet-assertions-5.7.0.csv*
