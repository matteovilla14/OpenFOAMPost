#!/bin/bash

# find and clean files from 'test' folder

exts=('*.png' '*.jpg' '*.svg' '*.eps' '*.pdf')
echo Cleaning 'test' directory...
echo

for ext in ${exts[@]}
do
    find test -type f -name $ext -delete
done

exit 0