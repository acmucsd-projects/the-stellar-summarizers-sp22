#!/bin/bash

# usage: use ffmpeg to convert videos from SumMe dataset into individual frames of jpgs

# create the root directory for all the frames
if [ ! -d "frames" ]
then
  mkdir ./frames
fi

for f in ./SumMe/videos/*.webm
do
  echo "Processing $f"
  dirname=$(echo $f | tr ' ' '_' | cut -d '/' -f 4 | cut -d '.' -f 1) 
  echo $dirname

  # if subdirectory does not exist
  if [ ! -d "./frames/$dirname" ]
  then
    echo "Creating directory ./frames/$dirname"
    mkdir ./frames/$dirname
  fi

  echo "Converting video and saving frames to ./frames$dirname"
  ffmpeg -i "$f" ./frames/$dirname/"img_%05d.jpg" > /dev/null
done