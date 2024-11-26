#!/bin/bash

script=graphene.py
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

python3 -m cProfile -o $profile $script $@
python3 stats.py $profile > $stats
