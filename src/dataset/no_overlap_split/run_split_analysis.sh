#!/bin/bash

source start.sh
python src/dataset/no_overlap_split/splitter.py
python src/dataset/no_overlap_split/analysis.py