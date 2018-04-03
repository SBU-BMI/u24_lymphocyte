#!/bin/bash

FILE=${1}

awk '
{
    slides[$1]=1;
    infil[$1" "$2]=$3;
}
END{
    for (s in slides) {
        print s, infil[s" azhao83"], infil[s" john.vanarnam"], infil[s" rebeccacydney"];
    }
}
' ${FILE}

exit 0
