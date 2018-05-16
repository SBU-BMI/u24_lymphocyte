#!/bin/bash

IMG=images

ls -l clicks | grep clicked | awk -F'clicked_|.txt' '{print $2}' > clicks/list_clicked.txt
ls -l clicks | grep ignored | awk -F'ignored_|.txt' '{print $2}' > clicks/list_ignored.txt

awk '
FILENAME=="clicks/list_ignored.txt"{
        igr[$1]=0;
}
FILENAME=="clicks/list_clicked.txt"{
        pos[$1]=0;
}
FILENAME=="images/info.txt"{
        if (!($1 in igr)) {
                if ($1 in pos)
                        n[$NF]++
                else
                        n[$NF] = n[$NF] + 0
        } else {
                n[$NF] = n[$NF] + 0
        }
}

END{
        for (x in n) {
                print x, n[x];
        }
}
' clicks/list_ignored.txt clicks/list_clicked.txt images/info.txt > clicks/groups.txt

exit 0
