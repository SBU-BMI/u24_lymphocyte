# /bin/bash

awk '$17==1{print $1,1;} $16==1{print $1,1.3} $15==1{print $1,1.8} $18==1{print $1,3}' image/label.txt | awk '{print $1"\t"$2}' > image/roundness.txt

exit 0
