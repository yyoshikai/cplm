

starts=(  0 20000 40000 60000  80000 100000 120000 140000 160000 180000 200000)
ends=(20000 40000 60000 80000 100000 120000 140000 160000 180000 200000 220000)

for i in {0..10}; do
    python preprocess/pdb_fragment.py --root-dir /workspace/cheminfodata/pdb/240101/mmCIF \
        --processname 240101 --tqdm --num-workers 26 --max-n-atom 20000 --range-min ${starts[i]} --range-sup ${ends[i]}
done
