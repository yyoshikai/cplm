

# finetune coord_follow
torchrun finetune.py --test --pretrain-name 250424_follow \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000

torchrun finetune2.py --test --pretrain-name 250424_follow \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test2 --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000
# git add . && git commit -m check_finetune_finetune2_follow

<<hist

# finetune
torchrun finetune.py --test --pretrain-name 241226_bucket_coord_heavy_random \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000

torchrun finetune2.py --test --pretrain-name 241226_bucket_coord_heavy_random \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test2 --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000
hist
