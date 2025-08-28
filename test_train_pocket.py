for coord_follow in [False, True]:
    # ref
    import sys, os
    import yaml
    from addict import Dict
    sys.path.append("/workspace/cplm")
    from src.data.pretrain import UniMolPocketDataset, ProteinDataset
    from src.data.coord_transform import CoordTransform
    from src.data.tokenizer import ProteinAtomTokenizer, FloatTokenizer

    org_dir = "/workspace/cplm/training/results/250619_mamba"
    with open(f"{org_dir}/config.yaml") as f:
        args = Dict(yaml.safe_load(f))
    args.mol_data = '/workspace/cheminfodata/unimol/ligands/train.lmdb'
    args.pocket_data = '/workspace/cheminfodata/unimol/pockets/train.lmdb'

    coord_transform = CoordTransform(args.seed, True, True, args.coord_noise_std)
    protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)
    coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    pocket_data = UniMolPocketDataset(args.pocket_data, idx_to_key='str')
    pocket_data = ProteinDataset(pocket_data, protein_atom_tokenizer, coord_tokenizer, coord_transform, atom_heavy=not args.no_pocket_atom_heavy, coord_heavy=args.pocket_coord_heavy, atom_h=args.pocket_atom_h, coord_h=args.pocket_coord_h, coord_follow_atom=coord_follow)
    vocs = pocket_data.vocs()

    items = [pocket_data[i] for i in range(3)]

    # mod
    import sys, os
    import yaml
    from addict import Dict
    sys.path.append("/workspace/cplm")
    from src.data.pretrain2 import UniMolPocketDataset, ProteinDataset
    from src.data.coord_transform import CoordTransform
    from src.data.tokenizer import ProteinAtomTokenizer, FloatTokenizer

    org_dir = "/workspace/cplm/training/results/250619_mamba"
    with open(f"{org_dir}/config.yaml") as f:
        args = Dict(yaml.safe_load(f))
    args.mol_data = '/workspace/cheminfodata/unimol/ligands/train.lmdb'
    args.pocket_data = '/workspace/cheminfodata/unimol/pockets/train.lmdb'

    coord_transform = CoordTransform(args.seed, True, True, args.coord_noise_std)
    protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)
    coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    pocket_data = UniMolPocketDataset(args.pocket_data, idx_to_key='str')
    pocket_data = ProteinDataset(pocket_data, protein_atom_tokenizer, coord_tokenizer, coord_transform, atom_heavy=not args.no_pocket_atom_heavy, coord_heavy=args.pocket_coord_heavy, atom_h=args.pocket_atom_h, coord_h=args.pocket_coord_h, coord_follow_atom=coord_follow)
    vocs0 = pocket_data.vocs()

    items0 = [pocket_data[i] for i in range(3)]

    # check
    assert vocs == vocs0
    assert all(item == item0 for item, item0 in zip(items, items0))
