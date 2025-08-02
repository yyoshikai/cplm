rm -r 250530_test

python generate.py --data-dir /workspace/cplm/preprocess/results/finetune/r4_all --sname 250521_g16_min_5000_a0000005 --step 10000 --index 250530_pm --max-len 1500 --token-per-batch 150000 --genname 250530_test --n-trial 1

path_base=250530_test/250530_pm/250521_g16_min_5000_a0000005/10000/info.csv
path=250530_base/250521_g16_min_5000_a0000005/10000/250530_pm/info.csv

if [ -e $path_base ] && [ -e $path ]; then
    sha_base=`sha1sum $path_base`
    sha=`sha1sum $path`
    sha_base=${sha_base% *}
    sha=${sha% *}

    if [ "$sha_base" = "$sha" ]; then
        cd /workspace/cplm
        git add .
        git commit -m check_reinforce_generation
    fi
fi