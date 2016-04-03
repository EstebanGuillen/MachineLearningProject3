#!/bin/bash
for aufile in $(find . -type f -name '*.au'); do
  ffmpeg -y -i "$aufile" "${aufile%.au}.wav"
done
