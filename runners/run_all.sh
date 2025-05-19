#!/bin/bash

if runners/run_ucr.sh; then
    echo "success"
    runners/run_tsb_our.sh
else
    echo "fail" >&2
    exit 1
fi