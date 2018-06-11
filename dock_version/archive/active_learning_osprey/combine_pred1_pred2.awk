BEGIN{
    w1 = 0;
    w2 = 1;
}

NR==FNR{
    lym[$1" "$2] = $3;
    nec[$1" "$2] = $4;
}

NR!=FNR{
    l1 = lym[$1" "$2];
    l2 = $3;
    #print $1, $2, (w1*l1 + w2*l2) / (w1 + w2), nec[$1" "$2];
    print $1, $2, (l1**w1 * l2**w2)**(1/(w1+w2)), nec[$1" "$2];
}

