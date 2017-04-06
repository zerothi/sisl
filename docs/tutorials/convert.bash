#!/bin/bash

for pdf in [0-9]*.pdf
do
    png=${pdf//pdf/png}
    if [ ! -e $png ]; then
	echo "Converting $pdf to $png"
	convert -density 300 $pdf $png
	optipng -o7 $png 2>/dev/null >/dev/null
    fi
done
	   
