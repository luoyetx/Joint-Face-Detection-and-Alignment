#!/usr/bin/env bash

ffmpeg -i ./build/test.mp4 -f mp3 -vn ./build/test.mp3

ffmpeg -i ./build/test.mp3 -i ./build/result.avi -f mp4 -y ./build/result.mp4
