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
function gototargetdir()
{
    # save current directory
    old_pwd="$PWD"
    
    # go to test directory (or its subdirectories)
    if [[ "$PWD" == "$OFPOST_TEST"* ]]
    then
        target_dir="$PWD"
    else
        target_dir="$OFPOST_TEST"
    fi 

    echo "Going to $target_dir directory..."
    cd "$target_dir"
}

function gotooldpwd()
{
    # go back to "old_pwd"
    cd "$old_pwd"
    unset old_pwd
}

function ofpost()
{
    # run program in test directory
    gototargetdir
    echo

    # export PYTHONPATH
    old_pythonpath="$PYTHONPATH"
    export PYTHONPATH="$PYTHONPATH":"$OFPOST_PATH/src"

    # run ofpost
    python3 -m ofpost "$@"
    
    # restore pythonpath and go to "old_pwd"
    export PYTHONPATH="$old_pythonpath"
    gotooldpwd
}

function ofclean()
{
    # clean up test directory
    gototargetdir
    echo "Cleaning up..."
    echo

    # delete files with certain extensions
    exts=('*.png' '*.jpg' '*.svg' '*.eps' '*.pdf')

    for ext in ${exts[@]}
    do
        find . -type f -name $ext -delete
    done

    gotooldpwd
}

return 0