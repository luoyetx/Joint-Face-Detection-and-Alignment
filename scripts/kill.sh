#!/usr/bin/env bash

kill $(ps aux | grep 'caffe' | awk '{print $2}')
