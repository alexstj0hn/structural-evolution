#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the complex log-likelihood for a list of PDB files using a pre-trained ESM model.

This script loads each PDB in turn, extracts its residue coordinates and 
wild-type sequences, and computes the log-likelihood of the entire complex 
given a specified target chain (or multiple chains). By default, it uses the 
wild-type sequence for that chain as the 'target sequence.' This procedure 
relies on the 'score_sequence_in_complex' function, which computes the 
log-likelihood of the complex from the perspective of the specified chain(s).

References:
    - Rives, A. et al. (2021). "Biological structure and function emerge from 
      scaling unsupervised learning to 250 million protein sequences."
      bioRxiv. https://doi.org/10.1101/622803
    - Meier, J. et al. (2021). "Language models enable zero-shot prediction of 
      the effects of mutations on protein function." 
      Advances in Neural Information Processing Systems, 34, 14931–14943.

Author: Your Name
Python Version: 3.10
"""

import argparse
import warnings
import torch
import os
import esm
import pandas as pd
from typing import List

# These utility imports assume you have the same 'multichain_util.py' and 'util.py' 
# modules as in your existing codebase. Adjust imports if needed.
from multichain_util import extract_coords_from_complex, score_sequence_in_complex
from util import load_structure, CoordBatchConverter
from recommend import get_model_checkpoint_path


def score_complexes(
    pdb_files: List[str],
    output_csv: str = "complex_scores.csv"
) -> pd.DataFrame:
    """
    For each PDB in 'pdb_files', compute the complex log-likelihood for each
    of the provided 'chain_ids' using its wild-type sequence. Returns a 
    pandas DataFrame and optionally saves it as CSV.

    :param pdb_files: List of paths to PDB or mmCIF files.
    :param chain_ids: List of chain identifiers (e.g. ['A'] or ['A','B']).
    :param output_csv: Path to save the resulting DataFrame as CSV.

    :return: A pandas DataFrame with columns:
        ["pdb_file", "chain", "complex_ll", "target_chain_ll"] 
        containing the log-likelihood of the entire complex from the perspective
        of the target chain and the target chain's own log-likelihood, respectively.
    """

    # ---------------------------
    # Load the ESM model once
    # ---------------------------
    print("Loading ESM model...")
    model_checkpoint_path = get_model_checkpoint_path('esm_if1_20220410.pt')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    # Create the batch converter
    batch_converter = CoordBatchConverter(alphabet)

    # ---------------------------
    # Score each PDB file
    # ---------------------------
    results = []
    for pdb_file in pdb_files:
        if not os.path.isfile(pdb_file):
            print(f"Warning: File '{pdb_file}' not found. Skipping.")
            continue

        # Load structure and extract coordinates + native sequences
        try:
            structure = load_structure(pdb_file)
            coords, native_seqs = extract_coords_from_complex(structure)
        except Exception as e:
            print(f"Error processing '{pdb_file}': {e}")
            continue
            
        # 'score_sequence_in_complex' can take multiple sequences at once. Here 
        # we pass just one entry in a list. We retrieve the [0]th entry from
        # the results since only one sequence is given.
        print(len(coords))
        print(native_seqs)
        print([native_seqs['A'][0]])
        ll_complex_list, _ = score_sequence_in_complex(
            model=model,
            alphabet=alphabet,
            coords=coords,
            native_seqs=native_seqs,
            target_chain_id='A',
            target_seq_list=[native_seqs['A']],
            batch_converter=batch_converter,
            device=device,
        )

        # Each returned list has length 1
        ll_complex = ll_complex_list[0]

        results.append({
            "pdb_file": pdb_file,
            "complex_ll": ll_complex,
        })

    # Build a DataFrame
    df = pd.DataFrame(results)

    # Optionally save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    return df


def main():
    """
    Parse command line arguments and compute the complex log-likelihood
    for a list of PDB files. Each file is scored for the specified chain(s).
    """
    parser = argparse.ArgumentParser(
        description="Compute complex log-likelihood for each PDB file using a pre-trained ESM model."
    )
    parser.add_argument(
        "--pdb_files", 
        type=str, 
        required=True,
        help="Comma-separated list of PDB/mmCIF files to process."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="complex_scores.csv", 
        help="Path to save the resulting CSV."
    )
    args = parser.parse_args()

    # Parse the lists
    pdb_list = [f.strip() for f in args.pdb_files.split(",")]

    # Compute the scores and save
    score_complexes(
        pdb_files=pdb_list, 
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
