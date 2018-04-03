NR==FNR{
    k = $1"\t"$3"\t"$4"\t"$5"\t"$6;
    h[k] = 1;
}

NR!=FNR{
    k = $1"\t"$3"\t"$4"\t"$5"\t"$6;
    if (k in h)
        print
}

