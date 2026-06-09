"""
260606 作成

under construction

"""

from collections.abc import Generator
from rdkit import Chem
from openbabel import openbabel as ob


class MolTokenizer:
    def encode(self, mol: Chem.Mol|ob.OBMol) -> list[str]:
        raise NotImplementedError
    
    def decode(self, tokens: list[str]) -> tuple[Chem.Mol, list[int], list[list[int]]]:
        """
        
        Returns
        -------
        mol: Output molecule
        atom_positions: list[int]
            atom i in mol corresponds to tokens[atom_positions[i]]
        coord_positions: list[int]
            coordinate of atom i in mol corresponds to token[coord_positions[i]]
        """

        raise NotImplementedError
    
    def generate_stream(self) -> Generator[tuple[bool, int, list[str]]]:
        """
        
        Yields
        ------
        is_remain: True if further generation required, False else ended
        pos: next position
        token_range: next token range
        """


def check_mol_tokenizer(mol_tokenizer: MolTokenizer, mols: list[Chem.Mol]):

    # encode -> decode


