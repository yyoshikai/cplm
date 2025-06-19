export CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun --nproc-per-node 1 train.py --studyname test --token-per-step 160000 \
    --mol-repeat 1 --pocket-repeat 5 --frag-repeat 0 --num-workers 28 \
    --mol-data /workspace/ssd/cheminfodata/unimol/ligands/train.lmdb \
    --pocket-data /workspace/ssd/cheminfodata/unimol/pockets/train.lmdb \
    --test --sdp-kernel FLASH --mamba \
    --pocket-coord-heavy --logtime --prefetch-factor 10 --coord-follow-atom