import esm
from tqdm import tqdm
import warnings
from multichain_util import extract_coords_from_complex, score_sequence_in_complex
from util import load_structure
import pandas as pd
from recommend import get_model_checkpoint_path
import numpy as np
from tqdm import tqdm
import torch 
from util import CoordBatchConverter

one_letter_code = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def simulate_annealing_batch(
    model,
    alphabet,
    coords,
    native_seqs,
    target_chain_id,
    initial_seqs,
    batch_converter,
    device,
    one_letter_code=None,
    n_steps=1000,
    T0=10.0,
    alpha=0.99,
    order=None,
):
    """
    Perform simulated annealing (SA) in parallel on `len(initial_seqs)` sequences.
    
    - We treat each sequence as an independent SA trajectory.
    - At iteration k, the temperature T = T0 * alpha^k.
    - We pick 1 random position for each sequence, mutate it, do a single batch pass.
    - For each sequence, if the new LL is higher, accept unconditionally; else accept with prob exp((LL_new - LL_old)/T).
    - Each time a sequence is accepted, we log it in the results.

    Args:
        initial_seqs: List of starting sequences (strings).
        n_steps: Number of SA steps
        T0: Starting temperature
        alpha: Exponential decay factor for temperature
        one_letter_code: possible amino acids to mutate to.
    Returns:
        results: a DataFrame with columns [seq_idx, step, old_seq, new_seq, old_ll, new_ll, accepted].
        final_seqs: the final sequences after SA completes.
        final_lls: final log-likelihoods.
    """
    if one_letter_code is None:
        one_letter_code = list("ACDEFGHIKLMNPQRSTVWY")

    num_seqs = len(initial_seqs)
    # Evaluate initial log-likelihoods
    ll_complex_list, ll_targetchain_list = score_sequence_in_complex(
        model, alphabet, coords, native_seqs, target_chain_id,
        initial_seqs, batch_converter, device, order=order
    )

    # We'll track the LL for each sequence as we go
    current_lls = np.array(ll_targetchain_list, dtype=np.float32)
    # Also track sequences in a list
    current_seqs = list(initial_seqs)

    # We’ll store acceptance info in a list of dicts (then convert to DataFrame).
    results_records = []

    for step in range(n_steps):
        T = T0 * (alpha ** step)

        # 1) pick one random position in each sequence, mutate it
        proposed_seqs = []
        mutated_positions = []
        mutated_residues = []

        for i in range(num_seqs):
            seq_list = list(current_seqs[i])
            pos = np.random.randint(0, len(seq_list))
            old_aa = seq_list[pos]
            new_aa = np.random.choice(one_letter_code)
            seq_list[pos] = new_aa
            proposed_seqs.append(''.join(seq_list))
            mutated_positions.append(pos)
            mutated_residues.append((old_aa, new_aa))

        # 2) score the new sequences in a batch
        ll_complex_list_new, ll_targetchain_list_new = score_sequence_in_complex(
            model, alphabet, coords, native_seqs, target_chain_id,
            proposed_seqs, batch_converter, device, order=order
        )

        new_lls = np.array(ll_targetchain_list_new, dtype=np.float32)

        # 3) Accept/reject each new seq
        for i in range(num_seqs):
            old_seq = current_seqs[i]
            new_seq = proposed_seqs[i]
            old_ll = current_lls[i]
            new_ll = new_lls[i]

            delta_ll = new_ll - old_ll

            # Accept with prob 1 if better, else exp(delta_ll / T)
            if (delta_ll > 0) or (np.random.rand() < np.exp(delta_ll / T)):
                accepted = True
                current_seqs[i] = new_seq
                current_lls[i] = new_ll
            else:
                accepted = False

            # Log an entry if accepted or not. You might choose only to log accepted, 
            # but let's log everything for clarity.
            results_records.append({
                'seq_idx': i,
                'step': step,
                'pos': mutated_positions[i],
                'old_res': mutated_residues[i][0],
                'new_res': mutated_residues[i][1],
                'old_seq': old_seq,
                'new_seq': new_seq,
                'old_ll': float(old_ll),
                'new_ll': float(new_ll),
                'delta_ll': float(delta_ll),
                'temp': T,
                'accepted': accepted
            })

    # 4) Return final sequences & LLs
    final_seqs = current_seqs
    final_lls = current_lls.tolist()

    df_results = pd.DataFrame(results_records)
    return df_results, final_seqs, final_lls

def main(pdb_file, chain, n_steps=1000):
    # 1) Load model & alphabet
    model_checkpoint_path = 'esm_if1_20220410.pt'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model, alphabet = esm.pretrained.load_model_and_alphabet(
            model_checkpoint_path
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    # 2) Create batch converter once
    batch_converter = CoordBatchConverter(alphabet)

    # 3) Load structure
    structure = load_structure(pdb_file)
    coords, native_seqs = extract_coords_from_complex(structure)

    # For example, define some initial sequences (wildtype or random).
    # Let's say we do SA on a batch of 5 sequences, each is just the wildtype for now:
    target_chain_id = chain
    wt_seq = native_seqs[target_chain_id]
    initial_seqs = [wt_seq]*5  # 5 copies of wildtype (or vary them if you like)

    # 4) Run Simulated Annealing
    df_results, final_seqs, final_lls = simulate_annealing_batch(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_id=target_chain_id,
        initial_seqs=initial_seqs,
        batch_converter=batch_converter,
        device=device,
        n_steps=n_steps,  # you can tune
        T0=10.0,          # initial temperature, tune
        alpha=0.99        # decay factor, tune
    )

    # 5) Save the results
    df_results.to_csv("simulated_annealing_log.csv", index=False)
    # final_seqs: your final solutions after SA
    # final_lls: the final log-likelihood for each sequence
    final_results_df = pd.DataFrame({
        "seq_idx": range(len(final_seqs)),
        "final_sequence": final_seqs,
        "final_ll": final_lls,
    })
    final_results_df.to_csv("simulated_annealing_final.csv", index=False)

    print("Simulated Annealing finished.")

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Score sequences based on a given structure.'
    )
    parser.add_argument(
        '--pdb_file', type=str,
        help='input filepath, either .pdb or .cif',
    )
    parser.add_argument(
        '--chain', type=str,
        help='chain id for the chain of interest', default='A',
    )

    args = parser.parse_args()
    main(args.pdb_file, args.chain)