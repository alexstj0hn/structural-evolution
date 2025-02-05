#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-phase simulated annealing for protein design with multiple target chains,
while only passing the first chain’s sequence to score_sequence_in_complex.

References:
    - Kirkpatrick et al. (1983). "Optimization by Simulated Annealing." 
      Science 220(4598): 671–680.
    - Press et al. (2007). "Numerical Recipes: The Art of Scientific Computing."

Author: Your Name
Python Version: 3.10
"""

import esm
from tqdm import tqdm, trange
import warnings
import numpy as np
import pandas as pd
import torch
import json  # To load and process the mutation options JSON file
import os

from multichain_util import extract_coords_from_complex, score_sequence_in_complex
from util import load_structure, CoordBatchConverter
from recommend import get_model_checkpoint_path

# Standard one-letter codes for the 20 canonical amino acids:
one_letter_code = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
    'T', 'V', 'W', 'Y'
]


def simulate_annealing_batch_two_phase(
    model,
    alphabet,
    coords,
    native_seqs,
    target_chain_ids,
    initial_seqs,
    batch_converter,
    device,
    one_letter_code=None,
    n_steps: int = 1000,
    T0: float = 0.0010,
    A_low: float = 0.3,
    A_high: float = 0.6,
    temp_increase_factor: float = 1.05,
    temp_decrease_factor: float = 0.95,
    adjust_every: int = 10,
    alpha: float = 0.99,
    order=None,
    mutation_options=None,
    max_mutations: int = -1,
    wildtype_seq=None
):
    """
    Conduct two-phase simulated annealing on multiple target chains, but only
    the first chain's sequence is passed to the scoring function. Hence, the 
    log-likelihood for each proposal is computed solely by providing the
    first chain’s sequence to 'score_sequence_in_complex'.

    This function simultaneously mutates multiple chains in each trajectory. 
    However, for the Metropolis accept/reject step, only the first chain’s
    sequence is given to the scoring function. If your scoring function
    internally computes the total complex log-likelihood, then passing a single
    chain’s ID and sequence may suffice for your pipeline.

    Arguments:
        model: ESM-IF1 or another protein language model.
        alphabet: Alphabet object from ESM.
        coords: Pre-extracted coordinate data for the complex.
        native_seqs (dict): Dictionary of chain_id -> wildtype sequence.
        target_chain_ids (list[str]): List of chains to mutate.
        initial_seqs (dict): { chain_id -> list[str] } where each list contains
            'n_init' starting sequences for that chain.
        batch_converter (CoordBatchConverter): Batch converter for scoring.
        device: A torch device, e.g. 'cuda' or 'cpu'.
        one_letter_code (list[str]): List of amino acid single-letter codes.
        n_steps (int): Total number of steps for simulated annealing.
        T0 (float): Initial temperature for adaptive phase.
        A_low (float): Lower bound for acceptance rate in adaptive phase.
        A_high (float): Upper bound for acceptance rate in adaptive phase.
        temp_increase_factor (float): Factor to increase T if acceptance too low.
        temp_decrease_factor (float): Factor to decrease T if acceptance too high.
        adjust_every (int): Interval (in steps) for adjusting T in adaptive phase.
        alpha (float): Multiplicative factor for classical-phase temperature cooling.
        order: An optional order parameter passed to the scoring function.
        mutation_options (dict): If provided, chain-specific restrictions:
            { chainID -> { position -> { residue -> weight, ... }, ... }, ... }
        max_mutations (int): Maximum mutated positions per chain. If -1, no limit.
        wildtype_seq (dict): { chainID -> wildtype_seq_string } to compare
            with the current mutated sequences.

    Returns:
        df_results (pd.DataFrame): Detailed log of each mutation step.
        final_seqs (list[dict]): List of dictionaries for each trajectory, each 
            mapping chain_id -> final sequence.
        final_lls (list[float]): The final log-likelihoods for each trajectory 
            (based on first chain scoring).
    """
    if one_letter_code is None:
        one_letter_code = list("ACDEFGHIKLMNPQRSTVWY")

    # Convert to a list, in case 'target_chain_ids' is not already.
    chain_list = list(target_chain_ids)

    # Number of initial sequences for each chain (we assume all chains have the same n_init).
    n_init = len(next(iter(initial_seqs.values())))

    # current_seqs: list of length 'n_init'; each entry is a dict {chainID -> sequence}
    current_seqs = []
    for i in range(n_init):
        chain_seq_dict = {}
        for ch in chain_list:
            chain_seq_dict[ch] = initial_seqs[ch][i]
        current_seqs.append(chain_seq_dict)

    num_seqs = len(current_seqs)

    # For scoring, we only pass the FIRST chain’s sequence to 'score_sequence_in_complex'
    scoring_chain_id = chain_list[0]

    # Build the list of sequences (strings) for that chain from each trajectory
    initial_seq_list_for_scoring = [traj[scoring_chain_id] for traj in current_seqs]

    # Score all initial trajectories using only the first chain.
    ll_complex_list, ll_targetchain_list = score_sequence_in_complex(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_id=scoring_chain_id,  # Only pass the first chain for scoring
        target_seq_list=initial_seq_list_for_scoring,
        batch_converter=batch_converter,
        device=device,
        order=order
    )
    current_lls = np.array(ll_complex_list, dtype=np.float32)

    results_records = []
    adaptive_steps = 0  # If desired, you can set e.g. n_steps//2 for adaptive
    classical_steps = n_steps - adaptive_steps

    T = T0
    accept_count_window = 0
    proposal_count_window = 0
    accepts = []

    pbar = trange(n_steps, desc="Simulated Annealing (Multi-chain, single-chain scoring)", leave=True)
    for step in pbar:
        # Make a deep(ish) copy of 'current_seqs' to propose updates
        proposed_seqs = [dict(d) for d in current_seqs]

        # In this variant, each trajectory attempts exactly ONE chain mutation
        # at each step, similarly to single-chain logic but repeated for each trajectory.
        mutated_info = []  # (chain_mut, pos_mut, old_aa, new_aa) for each trajectory

        for i in range(num_seqs):
            old_chain_seq_dict = current_seqs[i]

            # Randomly pick which chain to mutate
            chain_to_mutate = np.random.choice(chain_list)
            old_seq = list(old_chain_seq_dict[chain_to_mutate])

            # Determine valid positions to mutate
            if mutation_options is not None and chain_to_mutate in mutation_options:
                available_positions = list(mutation_options[chain_to_mutate].keys())
            else:
                available_positions = list(range(len(old_seq)))

            if max_mutations >= 0 and wildtype_seq is not None:
                wt_seq = wildtype_seq[chain_to_mutate]
                # Which positions are already mutated relative to WT?
                diff_positions = [p for p, aa in enumerate(old_seq) if aa != wt_seq[p]]
                if len(diff_positions) >= max_mutations:
                    can_mutate_positions = set(diff_positions).intersection(available_positions)
                else:
                    can_mutate_positions = set(available_positions)
            else:
                can_mutate_positions = set(available_positions)

            if not can_mutate_positions:
                # No valid mutation => no change
                mutated_info.append((chain_to_mutate, None, None, None))
                continue

            pos = np.random.choice(list(can_mutate_positions))

            # Decide the new residue
            if (mutation_options is not None 
                and chain_to_mutate in mutation_options 
                and pos in mutation_options[chain_to_mutate]):
                freq_dict = mutation_options[chain_to_mutate][pos]
                aas = list(freq_dict.keys())
                weights = list(freq_dict.values())
                wsum = sum(weights)
                weights = [w / wsum for w in weights]
                new_aa = np.random.choice(aas, p=weights)
            else:
                new_aa = np.random.choice(one_letter_code)

            old_aa = old_seq[pos]
            old_seq[pos] = new_aa
            mutated_seq = ''.join(old_seq)
            proposed_seqs[i][chain_to_mutate] = mutated_seq

            mutated_info.append((chain_to_mutate, pos, old_aa, new_aa))

        # Now compute the new log-likelihoods by again passing ONLY the first chain’s sequences
        proposed_seq_list_for_scoring = [p[scoring_chain_id] for p in proposed_seqs]

        ll_complex_new, ll_targetchain_new = score_sequence_in_complex(
            model=model,
            alphabet=alphabet,
            coords=coords,
            native_seqs=native_seqs,
            target_chain_id=scoring_chain_id,  # Only the first chain
            target_seq_list=proposed_seq_list_for_scoring,
            batch_converter=batch_converter,
            device=device,
            order=order
        )
        new_lls = np.array(ll_complex_new, dtype=np.float32)

        # Metropolis accept/reject
        accepted_count_step = 0
        for i in range(num_seqs):
            chain_mut, pos_mut, old_aa, new_aa = mutated_info[i]
            old_ll = current_lls[i]
            new_ll = new_lls[i]

            if pos_mut is None:
                # No valid mutation was proposed => skip
                accepted = False
                delta_ll = 0.0
                # revert the proposed changes
                proposed_seqs[i] = current_seqs[i]
                new_ll = old_ll
            else:
                delta_ll = new_ll - old_ll
                metropolis_threshold = np.exp(delta_ll / T)
                if delta_ll > 0 or (np.random.rand() < metropolis_threshold):
                    accepted = True
                    current_seqs[i] = proposed_seqs[i]
                    current_lls[i] = new_ll
                    accepted_count_step += 1
                else:
                    accepted = False
                    proposed_seqs[i] = current_seqs[i]
                    new_ll = old_ll

            # Log details
            results_records.append({
                'seq_idx': i,
                'step': step,
                'chain': chain_mut,
                'pos': pos_mut,
                'old_res': old_aa,
                'new_res': new_aa,
                'old_ll': float(old_ll),
                'new_ll': float(new_ll),
                'delta_ll': float(delta_ll),
                'temp': float(T),
                'accepted': accepted,
            })

        # Temperature update: adaptive first, then classical
        if step < adaptive_steps:
            accept_count_window += accepted_count_step
            proposal_count_window += num_seqs
            if (step + 1) % adjust_every == 0:
                acceptance_rate = accept_count_window / float(proposal_count_window)
                if acceptance_rate > A_high:
                    T *= temp_decrease_factor
                elif acceptance_rate < A_low:
                    T *= temp_increase_factor
                accept_count_window = 0
                proposal_count_window = 0
        else:
            T *= alpha

        # Progress bar updates
        accepts_step = [1]*accepted_count_step + [0]*(num_seqs - accepted_count_step)
        accepts.extend(accepts_step)
        window = accepts[-100:] if len(accepts) > 100 else accepts[:]
        recent_accept_rate = 100.0 * sum(window) / len(window) if window else 0.0

        mean_ll = float(np.mean(current_lls))
        pbar.set_description(
            f"SA Step {step} | {recent_accept_rate:.1f}% accepted (last 100) "
            f"| accepted {accepted_count_step}/{num_seqs} | T={T:.6f} | mean ll {mean_ll:.3f}"
        )

    # Final results
    final_seqs = current_seqs
    final_lls = current_lls.tolist()
    df_results = pd.DataFrame(results_records)

    print("Simulated Annealing complete. Final results per trajectory:")
    for i, (seq_dict, ll_val) in enumerate(zip(final_seqs, final_lls)):
        # Show only the first few characters of each chain for brevity
        chain_str = " | ".join(
            f"{ch}:{seq_dict[ch][:10]}...({len(seq_dict[ch])} AA)" for ch in seq_dict
        )
        print(f" - Traj {i}: LL={ll_val:.3f} | {chain_str}")

    return df_results, final_seqs, final_lls


def main(
    pdb_file,
    chain,
    n_steps=100,
    mutation_json_path=None,
    n_init=8,
    output_dir='.',
    max_mutations=-1,
    alpha=0.95
):
    """
    Main routine to load the ESM model and a protein structure, then run
    two-phase simulated annealing over multiple target chains. Only the first
    chain’s sequence is passed to 'score_sequence_in_complex' for scoring.

    Args:
        pdb_file (str): The filepath to the protein structure (PDB or mmCIF).
        chain (Union[str, list]): Chain identifier(s). If a string, it is 
            converted to a single-item list.
        n_steps (int): Total SA steps (sum of adaptive + classical).
        mutation_json_path (Optional[str]): JSON file specifying chain, position, 
            and frequency constraints on mutation.
        n_init (int): Number of parallel initial trajectories.
        output_dir (str): Directory to which results will be saved.
        max_mutations (int): Maximum number of mutated residues allowed 
            per chain, relative to wildtype. If -1, no limit is enforced.
        alpha (float): Multiplicative factor for classical-phase cooling.
    """
    print('Loading model...')
    model_checkpoint_path = get_model_checkpoint_path('esm_if1_20220410.pt')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    # Create the batch converter
    batch_converter = CoordBatchConverter(alphabet)

    # Load the structure and extract coordinates + native sequences
    structure = load_structure(pdb_file)
    coords, native_seqs = extract_coords_from_complex(structure)

    # Ensure 'chain' is a list
    if isinstance(chain, str):
        chain = [chain]

    # Prepare initial sequences: for each chain in 'chain', replicate the wildtype 'n_init' times
    initial_seqs = {
        chain_id: [native_seqs[chain_id]] * n_init
        for chain_id in chain
    }

    # Evaluate log-likelihood of the wildtype for the FIRST chain only (for reference)
    first_chain = chain[0]
    ll_complex_wt, ll_targetchain_wt = score_sequence_in_complex(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_id=first_chain,  # Only the first chain is passed
        target_seq_list=[native_seqs[first_chain]],
        batch_converter=batch_converter,
        device=device,
    )
    print(f"WT LL (complex) using chain {first_chain}: {ll_complex_wt[0]:.4f}")

    print('Running two-phase SA...')

    # If mutation options are provided, parse them
    mutation_options = None
    if mutation_json_path is not None:
        with open(mutation_json_path, 'r') as f:
            mutation_data = json.load(f)
        # Flatten the nested list (each sublist presumably contains one dictionary)
        entries = [item for sublist in mutation_data for item in sublist]

        # Build a dict of dicts: {chainID -> {pos -> {AA -> freq}}}
        mutation_options = {}
        for entry in entries:
            entry_chain = entry.get("chain")
            if entry_chain in chain:  # only if it matches one of our target chains
                pos = entry.get("position")
                freq = entry.get("frequency")
                if entry_chain not in mutation_options:
                    mutation_options[entry_chain] = {}
                mutation_options[entry_chain][pos] = freq

        print(f"Loaded mutation options from {mutation_json_path} for chains {chain}.")

    # Run the two-phase SA
    df_results, final_seqs, final_lls = simulate_annealing_batch_two_phase(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_ids=chain,
        initial_seqs=initial_seqs,
        batch_converter=batch_converter,
        device=device,
        n_steps=n_steps,
        one_letter_code=one_letter_code,
        mutation_options=mutation_options,
        max_mutations=max_mutations,
        wildtype_seq={c: native_seqs[c] for c in chain},
        alpha=alpha
    )

    # Save logs
    log_path = os.path.join(output_dir, "simulated_annealing_log.csv")
    df_results.to_csv(log_path, index=False)
    print(f"Saved detailed log to {log_path}")

    final_path = os.path.join(output_dir, "simulated_annealing_final.csv")
    print(final_seqs)
    final_results_df = pd.DataFrame({
        "seq_idx": range(len(final_seqs)),
        "light": [seq['A'] for seq in final_seqs],  # each entry is a dict of chain->seq
        "heavy": [seq['B'] for seq in final_seqs],
        "final_ll": final_lls,
    })
    final_results_df.to_csv(final_path, index=False)
    print(f"Saved final results to {final_path}")
    print("Simulated Annealing finished.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Two-phase simulated annealing for protein sequences with optional mutation constraints.'
    )
    parser.add_argument('--pdb_file', type=str, help='Input .pdb or .cif file')
    parser.add_argument('--chain', type=str, default='A', 
                        help='Chain identifier(s). If multiple, comma-separated e.g. "A,B"')
    parser.add_argument('--mutation_json', type=str, default=None,
                        help='Optional JSON file with mutation options.')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='Total number of SA steps.')
    parser.add_argument('--n_init', type=int, default=8,
                        help='Number of initial starting points (trajectories).')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory.')
    parser.add_argument('--max_mutations', type=int, default=-1,
                        help='Maximum mutated positions allowed per chain vs. wildtype. -1 = no limit.')
    parser.add_argument('--alpha', type=float, default=0.95,
                        help='Cooling factor for the classical phase.')
    args = parser.parse_args()

    # If user gave multiple chains in a single string, split them
    if ',' in args.chain:
        chain_list = [c.strip() for c in args.chain.split(',')]
    else:
        chain_list = [args.chain]

    main(
        pdb_file=args.pdb_file,
        chain=chain_list,
        n_steps=args.n_steps,
        mutation_json_path=args.mutation_json,
        n_init=args.n_init,
        output_dir=args.output_dir,
        max_mutations=args.max_mutations,
        alpha=args.alpha
    )
