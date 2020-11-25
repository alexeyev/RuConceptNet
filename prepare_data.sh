#!/bin/bash
echo "Starting preparing ru-conceptnet..."
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
mkdir ruconceptnet/data
echo "Filtering by Russian language..."
grep $'/ru/\S*\t\S*/ru/' conceptnet-assertions-5.7.0.csv > ruconceptnet/data/russian-conceptnet.tsv
echo "Building a compressed graph representation..."
python3 compressor.py
rm conceptnet-assertions-5.7.0.csv*
