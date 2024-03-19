#!/bin/bash

for input in "$PWD"/shortlist #longlist
do
  #echo $input
  (time ./a.out 22 33 < $input)
done