
<< hist
starts=(  0 20000 40000 60000  80000 100000 120000 140000 160000 180000 200000)
ends=(20000 40000 60000 80000 100000 120000 140000 160000 180000 200000 220000)

for i in {0..10}; do
    python preprocess/pdb_fragment.py --root-dir /workspace/cheminfodata/pdb/240101/mmCIF \
        --processname 240101 --tqdm --num-workers 26 --max-n-atom 20000 --range-min ${starts[i]} --range-sup ${ends[i]}
done

# 241119
python process_docking_types.py --input it2_tt_v1.3_10p20n_test0 2>preprocess/results/docking_types/it2_tt_v1.3_10p20n_test0.stderr
python process_docking_types.py --input it2_tt_v1.3_10p20n_test1 2>preprocess/results/docking_types/it2_tt_v1.3_10p20n_test1.stderr
notice
# 241119
python process_docking_types.py --input it2_tt_v1.3_0_train0 2>preprocess/results/docking_types/it2_tt_v1.3_0_train0.stderr
python process_docking_types.py --input it2_tt_v1.3_0_test0 2>preprocess/results/docking_types/it2_tt_v1.3_0_test0.stderr
notice
hist

# 241126 modified FinetuneDataset
python process_finetune_dataset.py --radius 4

