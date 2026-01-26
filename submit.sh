#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Arnav Mohnot,amohnot\nAnas Slassi,aslassi\nDhruv Srinivasan,dhruvs5" > submit/team.txt

# train model
python3 src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
python3 src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# submit document
cp -r docs submit/docs

# make zip file
zip -r submit.zip submit
