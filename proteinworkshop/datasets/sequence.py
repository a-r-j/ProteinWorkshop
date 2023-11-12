import os
import pathlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from graphein import verbose
from graphein.protein.tensor.io import protein_to_pyg
from graphein.protein.utils import read_fasta
from loguru import logger
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

try:
    import esm
except ImportError:
    logger.warning(
        "ESM not installed. If you are using a sequence dataset this will be required to fold structures. See: https://github.com/facebookresearch/esm#quickstart"
    )


verbose(False)


class SequenceDataset(Dataset):
    """Dataset class for working with Sequence Datasets. Provides utilities
    for batch folding and embedding with ESM(Fold)."""

    def __init__(
        self,
        fasta_file: Optional[str] = None,
        seq_representative: Optional[Union[str, os.PathLike]] = None,
        root: Optional[str] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        transform: Optional[List[Callable]] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        overwrite: bool = False,
        format: str = "fasta",
        use_embeddings: bool = True,
        use_structure: bool = True,
    ):
        self.root = root if root is not None else os.getcwd()
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.fasta_file = fasta_file
        self.sequences: Dict[str, str] = read_fasta(self.fasta_file)
        if self.sequences is None:
            raise ValueError("Must provide valid fasta file")

        if seq_representative is None:
            self.seq_representative = None
        elif seq_representative in self.sequences:
            self.seq_representative = (
                seq_representative,
                self.sequences[seq_representative],
            )
        else:
            raise ValueError(
                f"Representative sequence {seq_representative} not found in fasta file"
            )
        # convert dictionary to list for easier indexing
        self.sequences = list(self.sequences.items())
        self.overwrite = overwrite
        self.node_labels = node_labels
        self.graph_labels = graph_labels
        self.format = format
        self.embedding_dir = Path(self.root) / "embeddings"
        self.structure_dir = Path(self.root) / "structures"

        self.use_structure = use_structure
        self.use_embeddings = use_embeddings

        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        if not os.path.exists(self.structure_dir):
            os.makedirs(self.structure_dir)

        super().__init__(self.root, transform, pre_transform, pre_filter)

    @property
    def processed_dir(self) -> str:
        return str(self.structure_dir)

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{seq}.pdb" for seq in self.sequences]

    def process(self):
        if self.use_embeddings:
            self.embed()

        if self.use_structure:
            self.fold()

    def embed(
        self,
        embed_model: nn.Module = esm.pretrained.esm2_t33_650M_UR50D,
        layer_idx: int = 33,
        save: bool = True,
        overwrite: bool = False,
    ):
        """Embeds sequences using ESM-2 model. Embeddings are saved in the
        embeddings directory or returned."""

        model, alphabet = embed_model()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        # Filter out sequences that have already been embedded
        seqs_to_embed = self.sequences
        if save and not overwrite:
            seqs_to_embed = {
                seq_id: seq
                for seq_id, seq in seqs_to_embed
                if not os.path.exists(self.embedding_dir / f"{seq_id}.pt")
            }
            logger.info(
                f"Found {len(self.sequences) - len(seqs_to_embed)} existing embeddings in {self.embedding_dir}"
            )

        if save:
            logger.info(
                f"Creating ESM embeddings for {len(seqs_to_embed)} sequences in {self.embedding_dir}"
            )
        else:
            logger.info(
                f"Creating ESM embeddings for {len(seqs_to_embed)} sequences in memory"
            )
        if len(seqs_to_embed) == 0:
            return

        # Tokenize
        _, _, batch_tokens = batch_converter(seqs_to_embed)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        # Extract per-residue representations
        with torch.no_grad():
            results = model(
                batch_tokens, repr_layers=[layer_idx], return_contacts=True
            )
        token_representations = results["representations"][layer_idx]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token,
        # so the first residue is token 1.
        sequence_representations = [
            token_representations[i, 1 : tokens_len - 1].mean(0)
            for i, tokens_len in enumerate(batch_lens)
        ]
        if not save:
            return sequence_representations
        # Save representations and labels
        for i, (seq_id, seq) in enumerate(self.sequences):
            torch.save(
                sequence_representations[i],
                Path(self.embedding_dir) / f"{seq_id}.pt",
            )

    def fold(
        self, fold_model=esm.pretrained.esmfold_v1, overwrite: bool = False
    ):
        """
        Fold sequences using ESM-1b model.

        PDB files are saved in the structure directory or returned.
        """

        # If using representative sequence, only fold that one
        seqs_to_fold = (
            self.seq_representative
            if self.seq_representative is not None
            else self.sequences
        )

        if self.seq_representative is not None:
            logger.info(
                f"Folding representative sequence {self.seq_representative}"
            )
        else:
            logger.info(
                f"Folding {len(self.sequences)} sequences in {self.structure_dir}"
            )

        # If not overwrite, filter out sequences that have already been folded
        if not overwrite:
            seqs_to_fold = [
                (seq_id, seq)
                for seq_id, seq in self.sequences
                if not os.path.exists(
                    pathlib.Path(self.processed_dir) / f"{seq_id}.pt"
                )
            ]
            logger.info(
                f"Found {len(self.sequences) - len(seqs_to_fold)} existing structures in {self.structure_dir}"
            )

        # Instantiate model
        model = fold_model()
        # move model to GPU if available
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        # Predict structure and compute graph
        for seq_id, seq in tqdm(seqs_to_fold):
            with torch.no_grad():
                structure = model.infer_pdb(seq)
            with open(Path(self.structure_dir) / f"{seq_id}.pdb", "w") as f:
                f.write(structure)

            # Compute graph representation and save
            graph = protein_to_pyg(
                path=str(Path(self.structure_dir) / f"{seq_id}.pdb"),
                chain_selection="all",
                keep_insertions=True,
                store_het=False,
            )
            torch.save(self.processed_dir / f"{seq_id}.pt", graph)

    def len(self) -> int:
        return len(self.sequences)

    def get(self, idx: int) -> Data:
        """
        Returns PyTorch Geometric Data object for a given index, with structure if present.

        :param idx: Index to retrieve.
        :type idx: int
        :return: PyTorch Geometric Data object.
        """
        # check if structure exists
        seq_id, seq = self.sequences[idx]

        # Load embeddings
        if self.use_embeddings:
            emb = torch.load(self.embedding_dir / f"{seq_id}.pt")
            if not self.use_structure:
                return emb

        # Load structures
        if self.use_structure:
            struct = torch.load(self.structure_dir / f"{seq_id}.pt")
            if not self.use_embeddings:
                return struct

        if self.use_embeddings and self.use_structure:
            struct.esm = emb
            return struct

        # TODO load representative structure and modify sequence-related attributes
