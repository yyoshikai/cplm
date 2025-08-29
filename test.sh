

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# train
torchrun --nproc-per-node 1 train2.py --studyname test2 --token-per-step 160000 \
    --mol-repeat 1 --pocket-repeat 5 --frag-repeat 0 --num-workers 28 \
    --mol-data /workspace/ssd/cheminfodata/unimol/ligands/train.lmdb \
    --pocket-data /workspace/ssd/cheminfodata/unimol/pockets/train.lmdb \
    --test --sdp-kernel FLASH --seed 2\
    --pocket-coord-heavy --logtime --prefetch-factor 10 --coord-follow-atom

torchrun --nproc-per-node 1 train.py --studyname test --token-per-step 160000 \
    --mol-repeat 1 --pocket-repeat 5 --frag-repeat 0 --num-workers 28 \
    --mol-data /workspace/ssd/cheminfodata/unimol/ligands/train.lmdb \
    --pocket-data /workspace/ssd/cheminfodata/unimol/pockets/train.lmdb \
    --test --sdp-kernel FLASH --seed 2\
    --pocket-coord-heavy --logtime --prefetch-factor 10 --coord-follow-atom
# git add . && git commit -m check_train_train2_follow

<<hist

# train
torchrun --nproc-per-node 1 train2.py --studyname test2 --token-per-step 160000 \
    --mol-repeat 1 --pocket-repeat 5 --frag-repeat 0 --num-workers 28 \
    --mol-data /workspace/ssd/cheminfodata/unimol/ligands/train.lmdb \
    --pocket-data /workspace/ssd/cheminfodata/unimol/pockets/train.lmdb \
    --test --sdp-kernel FLASH --seed 2\
    --pocket-coord-heavy --logtime --prefetch-factor 10

torchrun --nproc-per-node 1 train.py --studyname test --token-per-step 160000 \
    --mol-repeat 1 --pocket-repeat 5 --frag-repeat 0 --num-workers 28 \
    --mol-data /workspace/ssd/cheminfodata/unimol/ligands/train.lmdb \
    --pocket-data /workspace/ssd/cheminfodata/unimol/pockets/train.lmdb \
    --test --sdp-kernel FLASH --seed 2\
    --pocket-coord-heavy --logtime --prefetch-factor 10 
# git add . && git commit -m check_train_train2

# finetune coord_follow
torchrun finetune.py --test --pretrain-name 250424_follow \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000

torchrun finetune2.py --test --pretrain-name 250424_follow \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test2 --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000
# git add . && git commit -m check_finetune_finetune2_follow

# finetune
torchrun finetune.py --test --pretrain-name 241226_bucket_coord_heavy_random \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000

torchrun finetune2.py --test --pretrain-name 241226_bucket_coord_heavy_random \
    --index-lmdb preprocess/results/finetune/r4_all/split/it2_0/train_idxs.lmdb  --finetune-save-dir preprocess/results/finetune/r4_all --num-workers 28 --loss-scale 6.25e-7 --token-per-step 100000 --duplicate overwrite --studyname test2 --clip-grad-value 1.0 --pin-memory --prefetch-factor 10 --pocket-coord-heavy \
    --scheduler warmup --lr 1e-3 --token-per-batch 50000
# git add . && git commit -m check_finetune_finetune2
hist
