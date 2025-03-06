#!/bin/bash

script=graphene.py
if [ $# -gt 0 ]; then
    script=$1
    shift
fi

# Base name
base=${script%.py}

# Stats
stats=$base.line_stats

echo "script: $script $@"
echo "saving stats to: $stats"

kernprof -l $script $@
python3 -m line_profiler $script.lprof > $stats
