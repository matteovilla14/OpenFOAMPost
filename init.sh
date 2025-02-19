#!/bin/bash

# initialize variables for developing purposes

# get script directory
shell="$(basename "$SHELL")"

if [ "$shell" = "bash" ]
then
    script_path="${BASH_SOURCE[0]}"
else
    script_path="$0"
fi

script_path="$(realpath "$script_path")" # get real location of this script
ofpost_path="$(dirname "$script_path")"
cd "$ofpost_path" ||
{ echo "ERROR: \"$ofpost_path\" directory not found."; return 1; }

# export environment variables
export OFPOST_PATH="$ofpost_path"
export OFPOST_TEST="$ofpost_path/test"
unset shell script_path ofpost_path

# create custom functions
function ofpost()
{
    # export PYTHONPATH
    old_pythonpath="$PYTHONPATH"
    export PYTHONPATH="$PYTHONPATH":"$OFPOST_PATH/src"

    # run ofpost
    python3 -m ofpost "$@"
    
    # restore pythonpath
    export PYTHONPATH="$old_pythonpath"
}

function ofpost-test()
{
    # run test for ofpost
    ofpost "$OFPOST_TEST" -c 2D "$@"
}

function ofpost-clean()
{
    # clean up test directory
    echo "Cleaning up $OFPOST_TEST..."
    echo

    # delete files with certain extensions
    exts=('*.png' '*.jpg' '*.svg' '*.eps' '*.pdf')

    for ext in ${exts[@]}
    do
        find "$OFPOST_TEST" -type f -name $ext -delete
    done
}

return 0