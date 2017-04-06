#!/bin/bash

for pdf in *.pdf
do
    convert -density 300 $pdf ${pdf//pdf/png}
done
	   
