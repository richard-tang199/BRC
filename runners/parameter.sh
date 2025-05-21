#!/bin/bash

if runners/parameter_ucr.sh; then
    echo "success"
    runners/parameter_tsb.sh
else
    echo "fail" >&2
    exit 1
fi
