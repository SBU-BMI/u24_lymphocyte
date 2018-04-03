#!/bin/bash

awk '{
    print $1, $2, $3, 0.0;
}' ./patch-level/${1} > ${1}

exit 0

