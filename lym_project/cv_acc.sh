#!/bin/bash

cat "$@" | awk 'match($(NF-1), "/"){h[$(NF-1)]+=$3; n[$(NF-1)]++} END{for(x in h){print x, h[x]/n[x], n[x]}}' | sort -k 3 -nr -k 2 -nr

exit 0
