
python ../runFactorGo.py \
    ./demo_n20_p1k.Zscore.gz  \
    ./demo_n20_p1k.SampleN.tsv  \
    -k 5 \
    --hyper 1e-5 1e-5 1e-5 1e-5 1e-5\
    --rate 50\
    --scale \
    -o ./demo\
    -p cpu
