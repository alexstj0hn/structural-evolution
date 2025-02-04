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
<<<<<<< HEAD

one_letter_code = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
=======
import json  # To load and process the mutation options JSON file

# Standard one-letter codes for the 20 amino acids.
one_letter_code = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
>>>>>>> 3ed5e5f (update_sa)


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
<<<<<<< HEAD
    T0=10.0,
    alpha=0.99,
=======
    T0=0.10,
    # Parameters for acceptance rate tuning:
    A_low=0.3,
    A_high=0.6,
    temp_increase_factor=1.05,
    temp_decrease_factor=0.95,
    adjust_every=10,  # Number of steps between temperature adjustments.
>>>>>>> 3ed5e5f (update_sa)
    order=None,
):
    """
<<<<<<< HEAD
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
=======
    Perform simulated annealing with an adaptive temperature schedule.
    
    If a dictionary of mutation options is provided (mapping residue positions to amino acid
    frequency dictionaries), mutations are restricted to the specified positions with the new amino acid
    chosen according to the provided weights. Otherwise, mutations are proposed at random positions
    and the new amino acid is selected uniformly from the provided alphabet.
    
    Args:
        model: The protein model.
        alphabet: The model’s alphabet.
        coords: Protein complex coordinates.
        native_seqs: Dictionary of native sequences extracted from the complex.
        target_chain_id (str): Chain identifier for the chain of interest.
        initial_seqs (List[str]): Initial sequences for the SA trajectories.
        batch_converter: Batch converter for processing input sequences.
        device: Computation device (e.g. "cuda" or "cpu").
        one_letter_code (List[str], optional): One-letter codes for amino acids.
        n_steps (int): Total number of simulated annealing steps.
        T0 (float): Initial temperature.
        A_low (float): Lower bound for the acceptance rate.
        A_high (float): Upper bound for the acceptance rate.
        temp_increase_factor (float): Multiplicative factor to increase T when acceptance is low.
        temp_decrease_factor (float): Multiplicative factor to decrease T when acceptance is high.
        adjust_every (int): Frequency (in steps) at which to adjust the temperature.
        order: Optional parameter for score_sequence_in_complex.
        mutation_options (dict, optional): A dictionary mapping residue positions (0-indexed)
            to dictionaries of amino acid weights.
    
    Returns:
        df_results (pd.DataFrame): Detailed log of each mutation step.
        final_seqs (List[str]): Final sequences after the SA process.
        final_lls (List[float]): Log-likelihoods for each final sequence.
>>>>>>> 3ed5e5f (update_sa)
    """
    if one_letter_code is None:
        one_letter_code = list("ACDEFGHIKLMNPQRSTVWY")

    num_seqs = len(initial_seqs)
<<<<<<< HEAD
    # Evaluate initial log-likelihoods
=======

    # 1) Evaluate initial log-likelihoods for each sequence.
>>>>>>> 3ed5e5f (update_sa)
    ll_complex_list, ll_targetchain_list = score_sequence_in_complex(
        model, alphabet, coords, native_seqs, target_chain_id,
        initial_seqs, batch_converter, device, order=order
    )

    # We'll track the LL for each sequence as we go
    current_lls = np.array(ll_targetchain_list, dtype=np.float32)
    # Also track sequences in a list
    current_seqs = list(initial_seqs)

<<<<<<< HEAD
    # We’ll store acceptance info in a list of dicts (then convert to DataFrame).
    results_records = []

    for step in range(n_steps):
        T = T0 * (alpha ** step)

        # 1) pick one random position in each sequence, mutate it
=======
    # Logging structure for results.
    results_records = []

    # Initialise temperature.
    T = T0

    # For tracking acceptance rate.
    accept_count_window = 0
    proposal_count_window = 0

    pbar = trange(n_steps, desc="Simulated Annealing (Adaptive)", leave=True)
    for step in pbar:

        # 1) Propose new sequences.
>>>>>>> 3ed5e5f (update_sa)
        proposed_seqs = []
        mutated_positions = []
        mutated_residues = []

        for i in range(num_seqs):
            seq_list = list(current_seqs[i])
<<<<<<< HEAD
            pos = np.random.randint(0, len(seq_list))
=======
            
            if mutation_options is not None:
                # Restrict mutations to positions specified in mutation_options.
                available_positions = list(mutation_options.keys())
                pos = np.random.choice(available_positions)
                # Obtain the weighted distribution for the selected position.
                weight_dict = mutation_options[pos]
                aas = list(weight_dict.keys())
                weights = list(weight_dict.values())
                total_weight = sum(weights)
                # Normalise weights.
                weights = [w / total_weight for w in weights]
                new_aa = np.random.choice(aas, p=weights)
            else:
                # Fallback: choose a random position and select a new amino acid uniformly.
                pos = np.random.randint(0, len(seq_list))
                new_aa = np.random.choice(one_letter_code)
            
>>>>>>> 3ed5e5f (update_sa)
            old_aa = seq_list[pos]
            new_aa = np.random.choice(one_letter_code)
            seq_list[pos] = new_aa
            proposed_seqs.append(''.join(seq_list))
            mutated_positions.append(pos)
            mutated_residues.append((old_aa, new_aa))

<<<<<<< HEAD
        # 2) score the new sequences in a batch
        ll_complex_list_new, ll_targetchain_list_new = score_sequence_in_complex(
            model, alphabet, coords, native_seqs, target_chain_id,
            proposed_seqs, batch_converter, device, order=order
=======
        # 2) Score the proposed sequences in a single batch.
        ll_complex_new, ll_targetchain_new = score_sequence_in_complex(
            model=model,
            alphabet=alphabet,
            coords=coords,
            native_seqs=native_seqs,
            target_chain_id=target_chain_id,
            target_seq_list=proposed_seqs,
            batch_converter=batch_converter,
            device=device,
            order=order
>>>>>>> 3ed5e5f (update_sa)
        )

<<<<<<< HEAD
        new_lls = np.array(ll_targetchain_list_new, dtype=np.float32)

        # 3) Accept/reject each new seq
=======
        # 3) Accept or reject each new sequence (Metropolis criterion).
        accepted_count_step = 0
>>>>>>> 3ed5e5f (update_sa)
        for i in range(num_seqs):
            old_seq = current_seqs[i]
            new_seq = proposed_seqs[i]
            old_ll = current_lls[i]
            new_ll = new_lls[i]

            delta_ll = new_ll - old_ll

<<<<<<< HEAD
            # Accept with prob 1 if better, else exp(delta_ll / T)
=======
>>>>>>> 3ed5e5f (update_sa)
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

<<<<<<< HEAD
    # 4) Return final sequences & LLs
=======
        accept_count_window += accepted_count_step
        proposal_count_window += num_seqs

        # Adjust the temperature every 'adjust_every' steps.
        if (step + 1) % adjust_every == 0:
            acceptance_rate = accept_count_window / float(proposal_count_window)
            if acceptance_rate > A_high:
                T *= temp_decrease_factor
                pbar.write(f"High acceptance rate ({acceptance_rate:.2f}); decreasing T to {T:.3f}")
            elif acceptance_rate < A_low:
                T *= temp_increase_factor
                pbar.write(f"Low acceptance rate ({acceptance_rate:.2f}); increasing T to {T:.3f}")
            else:
                pbar.write(f"Acceptance rate {acceptance_rate:.2f} within bounds; T remains {T:.3f}")

            accept_count_window = 0
            proposal_count_window = 0

        if accepted_count_step == 0:
            pbar.set_description(f"SA Step {step} | No accepts | T={T:.3f}")
        else:
            pbar.set_description(f"SA Step {step} | {accepted_count_step}/{num_seqs} accepted | T={T:.3f}")

>>>>>>> 3ed5e5f (update_sa)
    final_seqs = current_seqs
    final_lls = current_lls.tolist()

    df_results = pd.DataFrame(results_records)
<<<<<<< HEAD
    return df_results, final_seqs, final_lls

def main(pdb_file, chain, n_steps=1000):
    # 1) Load model & alphabet
    model_checkpoint_path = 'esm_if1_20220410.pt'
=======

    print("Simulated Annealing complete. Final sequences:")
    for i, (seq, ll_val) in enumerate(zip(final_seqs, final_lls)):
        print(f" - Sequence {i}: LL={ll_val:.3f} | {seq}")

    return df_results, final_seqs, final_lls


def main(pdb_file, chain, n_steps=100, mutation_json_path=None):
    """
    Main function to load the model, structure and mutation options, and then initiate simulated annealing.
    
    The mutation JSON file is expected to have the following nested structure:
    
        [
          [ { "chain": "A", "position": 27, "frequency": { ... } } ],
          [ { "chain": "A", "position": 28, "frequency": { ... } } ],
          ...
        ]
    
    Only entries matching the target chain are used. The output is a dictionary mapping positions (int)
    to the corresponding frequency dictionaries.
    
    Args:
        pdb_file (str): Filepath to the input structure (.pdb or .cif).
        chain (str): Chain identifier for the chain of interest.
        n_steps (int): Number of simulated annealing steps.
        mutation_json_path (str, optional): Filepath to the JSON file containing mutation options.
    
    """
    # 1) Load the protein model and alphabet.
    model_checkpoint_path = get_model_checkpoint_path('esm_if1_20220410.pt')
>>>>>>> 3ed5e5f (update_sa)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model, alphabet = esm.pretrained.load_model_and_alphabet(
            model_checkpoint_path
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

<<<<<<< HEAD
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
=======
    # 2) Create the batch converter.
    batch_converter = CoordBatchConverter(alphabet)

    # 3) Load the structure and extract coordinates.
    structure = load_structure(pdb_file)
    coords, native_seqs = extract_coords_from_complex(structure)

    # 4) Prepare initial sequences. Here we use 5 copies of the wildtype sequence.
    target_chain_id = chain
    wt_seq = native_seqs[target_chain_id]
    initial_seqs = [wt_seq] * 5

    # 5) Load and process the mutation options JSON file if provided.
    mutation_options = None
    if mutation_json_path is not None:
        with open(mutation_json_path, 'r') as f:
            mutation_data = json.load(f)
        # Flatten the nested list (each sublist contains one dictionary).
        entries = [item for sublist in mutation_data for item in sublist]
        # Filter entries for the target chain and build a dictionary mapping positions to frequency dictionaries.
        mutation_options = {}
        for entry in entries:
            if entry.get("chain") == target_chain_id:
                pos = entry.get("position")
                freq = entry.get("frequency")
                mutation_options[pos] = freq
        print(f"Loaded mutation options for chain {target_chain_id} from {mutation_json_path}")

    # 6) Run simulated annealing.
>>>>>>> 3ed5e5f (update_sa)
    df_results, final_seqs, final_lls = simulate_annealing_batch(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_id=target_chain_id,
        initial_seqs=initial_seqs,
        batch_converter=batch_converter,
        device=device,
<<<<<<< HEAD
        n_steps=n_steps,  # you can tune
        T0=10.0,          # initial temperature, tune
        alpha=0.99        # decay factor, tune
    )

    # 5) Save the results
=======
        n_steps=100,
        mutation_options=mutation_options
    )

    # 7) Save the results.
>>>>>>> 3ed5e5f (update_sa)
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
<<<<<<< HEAD
        help='chain id for the chain of interest', default='A',
=======
        help='Chain identifier for the chain of interest.',
        default='A'
    )
    parser.add_argument(
        '--mutation_json', type=str,
        help='Optional JSON file containing mutation options.',
        default=None
>>>>>>> 3ed5e5f (update_sa)
    )

    args = parser.parse_args()
    main(args.pdb_file, args.chain)