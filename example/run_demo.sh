
python runFactorGo.old.py \
    ./demo_n20_p1k.Zscore.gz  \
    ./demo_n20_p1k.SampleN.tsv  \
    -k 5 \
    --hyper 1e-5 1e-5 1e-5 1e-5 1e-5\
    --rate 20\
    --scale "True" \
    -o ./demo\
    -p cpu
