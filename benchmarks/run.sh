#!/bin/bash

script=graphene_100x100.py
if [ $# -gt 0 ]; then
    script=$1
    shift
fi

# Base name
base=${script%.py}

# Determine output profile
profile=$base.profile

# Stats
stats=$base.stats

python -m cProfile -o $profile $script $@
python stats.py $profile > $stats

