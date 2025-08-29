

export CUBLAS_WORKSPACE_CONFIG=:4096:8

<<hist
# reinforce_followはやっていないが, 面倒なので飛ばす

# reinforce
torchrun reinforce.py --finetune-name 250510_rft_coord --finetune-step 5000 \
    --finetune-save-dir preprocess/results/finetune/r4_all \
    --target qvina --max-step 10000 --record-opt-step 500\
    --num-workers 20 --num-score-workers 1 --studyname test --pin-memory --prefetch-factor 10 --pocket-coord-heavy --sdp-kernel FLASH --reward-scale sample_mean --generate-per-sample 2 --min-score ' -10' --ignore-error --batch-size 16 --tqdm-generate --tqdm

torchrun reinforce2.py --finetune-name 250510_rft_coord --finetune-step 5000 \
    --finetune-save-dir preprocess/results/finetune/r4_all \
    --target qvina --max-step 10000 --record-opt-step 500\
    --num-workers 20 --num-score-workers 1 --studyname test2 --pin-memory --prefetch-factor 10 --pocket-coord-heavy --sdp-kernel FLASH --reward-scale sample_mean --generate-per-sample 2 --min-score ' -10' --ignore-error --batch-size 16 --tqdm-generate --tqdm
# git add . && git commit -m check_reinforce_reinforce2

# train_follow
torchrun --nproc-per-node 1 train2.py --studyname test2 --token-per-step 160000 \
    --mol-repeat 1 --pocket-repeat 5 --frag-repeat 0 --num-workers 28 \
    --mol-data /workspace/ssd/cheminfodata/unimol/ligands/train.lmdb \
    --pocket-data /workspace/ssd/cheminfodata/unimol/pockets/train.lmdb \
    --test --sdp-kernel FLASH --seed 2 --max-step 5\
    --pocket-coord-heavy --logtime --prefetch-factor 10 --coord-follow-atom

torchrun --nproc-per-node 1 train.py --studyname test --token-per-step 160000 \
    --mol-repeat 1 --pocket-repeat 5 --frag-repeat 0 --num-workers 28 \
    --mol-data /workspace/ssd/cheminfodata/unimol/ligands/train.lmdb \
    --pocket-data /workspace/ssd/cheminfodata/unimol/pockets/train.lmdb \
    --test --sdp-kernel FLASH --seed 2\
    --pocket-coord-heavy --logtime --prefetch-factor 10 --coord-follow-atom
# git add . && git commit -m check_train_train2_follow

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
