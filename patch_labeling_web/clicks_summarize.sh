#!/bin/bash

IMG=images

ls -l clicks | grep clicked | awk -F'clicked_|.txt' '{print $2}' > clicks/list_clicked.txt
ls -l clicks | grep ignored | awk -F'ignored_|.txt' '{print $2}' > clicks/list_ignored.txt

exit 0
