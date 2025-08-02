cd /workspace/cplm/pocket_conditioned_generation/finetune

rm -r 250802_base/250530_pm/250125_coord_heavy_random/84000
python generate.py --index 250530_pm --max-len 1500 --token-per-batch 150000 --sname 250125_coord_heavy_random --step 84000 --genname 250802_base

sha_base=`sha1sum 250802_base/250530_pm/250125_coord_heavy_random_base/84000/info.csv`
sha=`sha1sum 250802_base/250530_pm/250125_coord_heavy_random/84000/info.csv`
sha_base=${sha_base% *}
sha=${sha% *}

if [ "$sha_base" = "$sha" ]; then
    cd /workspace/cplm
    git add .
    git commit -m check_finetune_generation
fi
