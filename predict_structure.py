import sys, os, math
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, StackDataset
from torch.nn.utils.rnn import pad_sequence
from openbabel.openbabel import OBMol, OBResidueIter, OBResidueAtomIter

from src.utils.logger import get_logger, add_file_handler, set_third_party_logger
from src.utils.rdkit import ignore_rdkit_warning
from src.data import WrapDataset, index_dataset, TensorDataset
from src.data.datasets.pdb import PDBUniMolRandomDataset
from src.data.protein import ProteinTokenizeDataset
from src.data.tokenizer import SentenceDataset, VocEncoder, TokenEncodeDataset, TokenSplitDataset
from src.train import get_model
from src.evaluate import parse_coord_tokens

def get_ref(raw_idx):
    out_path = f"predict_structure/reference/{raw_idx}.csv"
    if os.path.exists(out_path): return
    protein: OBMol = PDBUniMolRandomDataset('valid')[raw_idx]
    atom_data = defaultdict(list)
    for residue_idx, residue in enumerate(OBResidueIter(protein)):
        amino = residue.GetName()
        for atom in OBResidueAtomIter(residue):
            atom_data['residue_idx'].append(residue_idx)
            atom_data['amino'].append(amino)
            atom_data['atom_name'].append(residue.GetAtomID(atom).strip())
            atom_data['X'].append(atom.GetX())
            atom_data['Y'].append(atom.GetY())
            atom_data['Z'].append(atom.GetZ())
    df = pd.DataFrame(atom_data)
    df.to_csv(out_path, index=False)

def to_graph(out_name):
    sys.path.append("/workspace")
    from tools.graph import get_grid, COLORS2, tightargs

    out_dir = f"predict_structure/results/{out_name}"
    df_info = pd.read_csv(f"{out_dir}/info.csv", index_col=0)
    for raw_idx in df_info['raw_idx']:
        get_ref(raw_idx)
    df_ref = [pd.read_csv(f"predict_structure/reference/{raw_idx}.csv") for raw_idx in df_info['raw_idx']]
    for mol_idx, df in enumerate(df_ref):
        df['residue_idx'] = [f'{mol_idx}_{res_idx}' for res_idx in df['residue_idx']]
    ns_atom = [len(df) for df in df_ref]
    df_ref = pd.concat(df_ref)
    unique_atom_idxs = np.sort(np.unique(df['residue_idx'].values, return_index=True)[1])
    df_residue_ref = df_ref.iloc[unique_atom_idxs]
    n_residue = len(df_residue_ref)
    
    coords = [np.load(f"{out_dir}/coords/{data_idx}.npy") for data_idx in df_info.index] # [data_idx, atom_idx, axis]
    n_coord = len(coords)
    coords = np.concatenate(coords, axis=0) # [atom_idx, axis]

    for amino in np.sort(np.unique(df_ref['amino'])):
        if not amino.isalpha(): continue
        amino_mask = df_ref['amino'] == amino
        df_atom = df_ref[amino_mask]
        residue_size = np.sum(df_atom['residue_idx'] == df_atom['residue_idx'][0])
        n_residue_amino = len(df_atom) // residue_size
        atom_names = df_atom['atom_name']
        assert np.all(atom_names.reshape(-1, residue_size) == atom_names[np.newaxis, :residue_size])
        ref_coord = df_atom[['X', 'Y', 'Z']].values.reshape(n_residue_amino, residue_size, 3)
        coords = coords[amino_mask].reshape(n_coord, n_residue_amino, residue_size, 3)
        coords = np.concatenate([ref_coord[np.newaxis], coords], axis=0) # [output, res, atom, axis]
        matrix = np.sqrt(np.sum((coords[:,:,np.newaxis]-coords[:,:,:,np.newaxis])**2, axis=-1)) # [output, res, atom, atom]

        fig, axs = get_grid(residue_size**2, 4, 3, residue_size, batched=True)
        xs = np.where(df_residue_ref['amino'] == amino)[0]
        for i_atom, j_atom in itr.product(range(residue_size), range(residue_size)):
            ax = axs[i_atom][j_atom]
            if i_atom == j_atom:
                ax.axis('off')
                continue
            for i_out in range(n_coord+1):
                ys = matrix[i_out, :, i_atom, j_atom]
                if i_out == 0:
                    ax.plot(xs, ys, color='slategray', label='Reference', ms=5, marker='o', zorder=-1)
                else:
                    ax.plot(xs, ys, color=COLORS2[i_out-1], label=f"Trial #{i_out-1}", ms=5, marker='o', ls='')
            max_y = np.max(matrix[:,:,i_atom,j_atom])
            ax.set_xlim(-0.5, n_residue-0.5)
            ax.set_xlabel('Residue')
            ax.set_ylim(0, max_y*1.02)
            ax.set_ylabel('Distance')
            if i_atom == n_atom-1 and j_atom == n_atom-2:
                ax.legend(loc=(1.02, 0))
        fig.savefig(f"predict_structure/graph/{out_name}/{amino}.png", **tightargs)
        plt.close(fig)

if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument('--studyname', required=True)
    parser.add_argument('--opt', type=int)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--tqdm', action='store_true')
    parser.add_argument('--genname')
    parser.add_argument('--n-trial', type=int, default=5)
    parser.add_argument('--n-protein', type=int, default=5)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--max-prompt-len', type=int, default=math.inf)
    args = parser.parse_args()



    # environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = f"predict_structure/results/{args.genname or ''}/{args.studyname}/{args.opt}"
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{out_dir}/predict_structure.log")
    token_logger = get_logger("tokens")
    token_logger.propagate = False
    add_file_handler(token_logger, f"{out_dir}/tokens.log")
    token_logger.debug(f"[step][batch_idx][batch_index]=")
    ignore_rdkit_warning()
    set_third_party_logger()

    train_dir = f"training/results/{args.studyname}"
    targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
    assert targs.PDBUniMolRandom > 0
    assert not targs.coord_follow_atom

    # dataset
    class ProteinSizeDataset(WrapDataset[int]):
        def __init__(self, protein_data: WrapDataset[OBMol]):
            super().__init__(protein_data)
        
        def __getitem__(self, idx: int):
            protein: OBMol = self.dataset[idx]
            return protein.NumAtoms()

    protein = PDBUniMolRandomDataset('valid')
    raw_idxs, protein = index_dataset(protein)
    subset_idxs = [0]+np.random.default_rng(args.sample_seed).choice(len(protein), args.n_protein-1).tolist()
    protein = Subset(protein, subset_idxs)
    raw_idxs = Subset(raw_idxs, subset_idxs)
    protein_size = ProteinSizeDataset(protein)
    token = ProteinTokenizeDataset(protein, heavy_atom=not targs.no_pocket_heavy_atom, heavy_coord=not targs.no_pocket_heavy_coord, h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord, coord_follow_atom=targs.coord_follow_atom, atom_order=getattr(targs, 'pocket_atom_order', False), coord_range=targs.coord_range)
    atom_token = TokenSplitDataset(token, '[XYZ]')
    sentence = SentenceDataset('[POCKET]', atom_token, '[XYZ]')
    vocs = sentence.vocs() | {'[END]'}
    token, position = sentence.untuple()
    voc_encoder = VocEncoder(vocs)
    pad_token = voc_encoder.pad_token
    end_token = voc_encoder.voc2i['[END]']
    token = TokenEncodeDataset(token, voc_encoder)
    position = TensorDataset(position, torch.long)
    data_idx, token = index_dataset(token)
    data = StackDataset(data_idx, raw_idxs, token, position, protein_size)
    sampler = itr.chain(*itr.repeat(range(len(data)), args.n_trial))
    # model
    model = get_model(targs, voc_encoder, f"{train_dir}/models/{args.opt}.pth", device)
    loader = DataLoader(data, args.batch_size, sampler=itr.chain(*itr.repeat(range(len(data)), args.n_trial)),
            num_workers=args.num_workers, collate_fn=lambda x: x)
    errors = [[] for _ in range(len(data))]
    coords = [[] for _ in range(len(data))]

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            batch_data_idxs, batch_raw_idxs, tokens, positions, ns_atom = zip(*batch)
            for data_idx, token in zip(batch_data_idxs, tokens):
                token_size = len(token)
                if token_size > args.max_prompt_len:
                    errors[data_idx].append('LARGE_PROMPT')
            batch = [data for data, token in zip(batch, tokens) if len(token) <= args.max_prompt_len]
            if len(batch) == 0:
                logger.info(f"step[{step}] All prompts were too large.")
                continue
            batch_data_idx, batch_raw_idxs, tokens, positions, ns_atom = zip(*batch)
            tokens = pad_sequence(tokens)
            positions = pad_sequence(positions)
            max_len = max(ns_atom)*6+1

            L, B = tokens.shape
            tokens = tokens.to(device)
            positions = positions.to(device)
            
            logger.info(f"step[{step}] generating {batch_raw_idxs} ...")
            outputs = model.generate(tokens, '[END]', max_len, pad_token, tqdm=args.tqdm)
            outputs = [out.cpu().numpy() for out in outputs]

            # Log tokens
            if step == 0:
                for batch_idx, input, output in zip(batch_data_idxs, tokens.T, outputs):
                    token_logger.debug(f"[{step}][{batch_idx}]Input={voc_encoder.decode(input)}")
                    token_logger.debug(f"[{step}][{batch_idx}]Output={voc_encoder.decode(output)}")

            logger.debug(f"step[{step}] parsing...")
            for i, (data_idx, n_atom, output) in enumerate(zip(batch_data_idxs, ns_atom, outputs)):
                if output[-1] != end_token:
                    error = 'COORD_SIZE'
                words = voc_encoder.decode(output[:-1])
                error, coord = parse_coord_tokens(words)
                if coord is None:
                    coord = np.full((n_atom, 3), np.nan)
                if len(coord) < n_atom:
                    coord = np.stack([coord, np.full((n_atom-len(coord), 3), np.nan)])
                errors[data_idx].append(error)
                coords[data_idx].append(coord)
    df = pd.DataFrame(errors, index=range(1, args.n_trial+1))
    df['raw_idx'] = raw_idxs
    df.to_csv(f"{out_dir}/info.csv")
    for data_idx, coord in enumerate(coords):
        coord = np.stack(coords, axis=0)
        np.save(f"{out_dir}/coords/{data_idx}.npy", coord)

    # to graph
    for raw_idx in raw_idxs:
        get_ref(raw_idx)




