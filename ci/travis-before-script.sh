#!/bin/bash

if [[ $TRAVIS_OS_NAME != 'osx' ]]; then
    export DISPLAY=:99.0
    sh -e /etc/init.d/xvfb start
    sleep 2
fi
export MPLBACKEND=Agg
