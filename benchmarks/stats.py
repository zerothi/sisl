# Script for analysing profile scripts created by the
# cProfile module.
import sys
import pstats

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    raise ValueError('Must supply a profile file-name')

stat = pstats.Stats(fname)

# We sort against total-time
stat.sort_stats('tottime')
# Only print the first 20% of the routines.
stat.print_stats('sisl', 0.2)
