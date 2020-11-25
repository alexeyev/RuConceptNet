#!/bin/bash
echo "Starting preparing ru-conceptnet..."
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
mkdir data
echo "Filtering by Russian language..."
grep $'/ru/\S*\t\S*/ru/' conceptnet-assertions-5.7.0.csv > data/russian-conceptnet.tsv
echo "Building a compressed graph representation..."
python3 ruconceptnet/compressor.py
rm conceptnet-assertions-5.7.0.csv*
