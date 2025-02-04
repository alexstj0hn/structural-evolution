import esm
from tqdm import tqdm, trange
import warnings
from multichain_util import extract_coords_from_complex, score_sequence_in_complex
from util import load_structure
import pandas as pd
from recommend import get_model_checkpoint_path
import numpy as np
import torch
from util import CoordBatchConverter
import json  # To load and process the mutation options JSON file

# Standard one-letter codes for the 20 amino acids.
one_letter_code = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def simulate_annealing_batch_two_phase(
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
    T0=0.010,
    # Adaptive-phase acceptance tuning parameters:
    A_low=0.3,
    A_high=0.6,
    temp_increase_factor=1.05,
    temp_decrease_factor=0.95,
    adjust_every=10,  # Frequency of temperature adjustments in the adaptive phase.
    # Classical-phase monotonic cooling parameters:
    classical_decrease_factor=0.99,
    order=None,
    mutation_options: dict = None
):
    """
    Perform two-phase simulated annealing:
      (1) An adaptive phase (acceptance-rate-based temperature adjustments).
      (2) A classical phase with a monotonically decreasing temperature schedule.

    By default, n_steps is evenly split between the adaptive and classical phases,
    with no acceptance-based temperature tuning after the initial adaptive phase.

    If a dictionary of mutation options is provided (mapping residue positions to amino acid
    frequency dictionaries), mutations are restricted to the specified positions with the new amino acid
    chosen according to the provided weights. Otherwise, mutations are proposed at random positions
    and the new amino acid is selected uniformly from the provided alphabet.

    References:
        - Kirkpatrick, S. et al. “Optimization by Simulated Annealing.” (Science, 1983).
    
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
        T0 (float): Initial temperature for the adaptive phase.
        A_low (float): Lower bound for the acceptance rate in the adaptive phase.
        A_high (float): Upper bound for the acceptance rate in the adaptive phase.
        temp_increase_factor (float): Factor to increase T when acceptance is low (adaptive phase).
        temp_decrease_factor (float): Factor to decrease T when acceptance is high (adaptive phase).
        adjust_every (int): Frequency (in steps) for acceptance-based T adjustments in the adaptive phase.
        classical_decrease_factor (float): Monotonic decrease factor in the classical phase.
        order: Optional parameter for score_sequence_in_complex.
        mutation_options (dict, optional): A dictionary mapping residue positions (0-indexed)
            to dictionaries of amino acid weights.

    Returns:
        df_results (pd.DataFrame): Detailed log of each mutation step.
        final_seqs (List[str]): Final sequences after the SA process.
        final_lls (List[float]): Log-likelihoods for each final sequence.
    """
    if one_letter_code is None:
        one_letter_code = list("ACDEFGHIKLMNPQRSTVWY")

    num_seqs = len(initial_seqs)

    # 1) Evaluate initial log-likelihoods for each sequence.
    ll_complex_list, ll_targetchain_list = score_sequence_in_complex(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_id=target_chain_id,
        target_seq_list=initial_seqs,
        batch_converter=batch_converter,
        device=device,
        order=order
    )
    current_lls = np.array(ll_complex_list, dtype=np.float32)
    current_seqs = list(initial_seqs)

    # Logging structure for results.
    results_records = []

    # Split the total steps between adaptive and classical phases.
    adaptive_steps = n_steps // 2
    classical_steps = n_steps - adaptive_steps

    # Initialise temperature.
    T = T0

    # For tracking acceptance rate in the adaptive phase.
    accept_count_window = 0
    proposal_count_window = 0

    # We will run a single loop from 0..n_steps, but with different update rules.
    pbar = trange(n_steps, desc="Simulated Annealing (Two-Phase)", leave=True)
    accepts = []
    for step in pbar:

        # 1) Propose new sequences.
        proposed_seqs = []
        mutated_positions = []
        mutated_residues = []
        for i in range(num_seqs):
            seq_list = list(current_seqs[i])
            
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
            
            old_aa = seq_list[pos]
            seq_list[pos] = new_aa

            proposed_seqs.append(''.join(seq_list))
            mutated_positions.append(pos)
            mutated_residues.append((old_aa, new_aa))

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
        )
        new_lls = np.array(ll_complex_new, dtype=np.float32)

        # 3) Accept or reject each new sequence (Metropolis criterion).
        accepted_count_step = 0
        for i in range(num_seqs):
            old_seq = current_seqs[i]
            new_seq = proposed_seqs[i]
            old_ll = current_lls[i]
            new_ll = new_lls[i]
            delta_ll = new_ll - old_ll

            if (delta_ll > 0) or (np.random.rand() < np.exp(delta_ll / T)):
                accepted = True
                current_seqs[i] = new_seq
                current_lls[i] = new_ll
                accepted_count_step += 1
            else:
                accepted = False

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
                'temp': float(T),
                'accepted': accepted
            })

        # =====================
        # Phase 1: Adaptive
        # =====================
        if step < adaptive_steps:
            accept_count_window += accepted_count_step
            proposal_count_window += num_seqs

            # Adjust the temperature every 'adjust_every' steps during the adaptive phase.
            if (step + 1) % adjust_every == 0:
                acceptance_rate = accept_count_window / float(proposal_count_window)
                if acceptance_rate > A_high:
                    T *= temp_decrease_factor
                    pbar.write(f"[Adaptive] High acceptance rate ({acceptance_rate:.2f}); decreasing T to {T:.3f}")
                elif acceptance_rate < A_low:
                    T *= temp_increase_factor
                    pbar.write(f"[Adaptive] Low acceptance rate ({acceptance_rate:.2f}); increasing T to {T:.3f}")
                else:
                    pbar.write(f"[Adaptive] Acceptance rate {acceptance_rate:.2f} within bounds; T remains {T:.3f}")

                accept_count_window = 0
                proposal_count_window = 0

        # =====================
        # Phase 2: Classical (monotonically decreasing)
        # =====================
        else:
            # Once we enter the classical phase, we do not adapt T based on acceptance.
            # Instead we simply apply a monotonic cooling schedule each step.
            T *= classical_decrease_factor

        if np.sum(accepts) == 0:
            recent_accepts = 0
        elif len(accepts) < 100:
            recent_accepts = 100*(np.sum(accepts)/len(accepts))
        else:
            recent_accepts = np.sum(accepts[-100:])

        accepts.extend(([1] * accepted_count_step) + ([0] * (num_seqs - accepted_count_step)))
        
        if accepted_count_step == 0:
            pbar.set_description(f"SA Step {step} | {recent_accepts:.1f}% accepted in last 100 steps | 0/{num_seqs} accepted | T={T:.8f} | current mean ll {np.mean(current_lls):.5f}")
        else:
            pbar.set_description(f"SA Step {step} | {recent_accepts:.1f}% accepted in last 100 steps | {accepted_count_step}/{num_seqs} accepted | T={T:.8f} | current mean ll {np.mean(current_lls):.5f}")

    final_seqs = current_seqs
    final_lls = current_lls.tolist()
    df_results = pd.DataFrame(results_records)

    print("Simulated Annealing complete. Final sequences:")
    for i, (seq, ll_val) in enumerate(zip(final_seqs, final_lls)):
        print(f" - Sequence {i}: LL={ll_val:.3f} | {seq}")

    return df_results, final_seqs, final_lls


def main(pdb_file, chain, n_steps=100, mutation_json_path=None, n_init=8):
    """
    Main function to load the model, structure and mutation options, and then initiate
    a two-phase simulated annealing approach.
    
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
        n_steps (int): Number of total SA steps (divided between adaptive and classical phases).
        mutation_json_path (str, optional): Filepath to the JSON file containing mutation options.
        n_init (int): Number of initial starting sequences.
    """
    # 1) Load the protein model and alphabet.
    print('Loading model')
    model_checkpoint_path = get_model_checkpoint_path('esm_if1_20220410.pt')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    # 2) Create the batch converter.
    batch_converter = CoordBatchConverter(alphabet)

    # 3) Load the structure and extract coordinates.
    structure = load_structure(pdb_file)
    coords, native_seqs = extract_coords_from_complex(structure)

    # 4) Prepare initial sequences. Here we use n_init copies of the wildtype sequence.
    target_chain_id = chain
    wt_seq = native_seqs[target_chain_id]
<<<<<<< HEAD
    initial_seqs = [wt_seq] * n_init
=======
    initial_seqs = [wt_seq] * 16
>>>>>>> 4edbf1c460dad3cbe95306d5065142517646bad1

    # 4.1) Evaluate initial log-likelihood for the wildtype sequence (for reference).
    ll_complex_wt, ll_targetchain_wt = score_sequence_in_complex(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_id=target_chain_id,
        target_seq_list=[wt_seq],
        batch_converter=batch_converter,
        device=device,
    )
    print(f"WT LL (complex): {ll_complex_wt[0]:.4f}")

    print('Running two-phase SA...')
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

    # 6) Run the two-phase simulated annealing.
    df_results, final_seqs, final_lls = simulate_annealing_batch_two_phase(
        model=model,
        alphabet=alphabet,
        coords=coords,
        native_seqs=native_seqs,
        target_chain_id=target_chain_id,
        initial_seqs=initial_seqs,
        batch_converter=batch_converter,
        device=device,
        n_steps=n_steps,
        mutation_options=mutation_options
    )

    # 7) Save the results.
    df_results.to_csv("simulated_annealing_log.csv", index=False)
    final_results_df = pd.DataFrame({
        "seq_idx": range(len(final_seqs)),
        "final_sequence": final_seqs,
        "final_ll": final_lls,
    })
    final_results_df.to_csv("simulated_annealing_final.csv", index=False)

    print("Simulated Annealing finished.")


if __name__ == '__main__':
    import argparse

    # Set up the argument parser.
    parser = argparse.ArgumentParser(
        description='Two-phase simulated annealing for protein sequences.'
    )
    parser.add_argument(
        '--pdb_file', type=str,
        help='Input filepath (either .pdb or .cif).'
    )
    parser.add_argument(
        '--chain', type=str,
        help='Chain identifier for the chain of interest.',
        default='A'
    )
    parser.add_argument(
        '--mutation_json', type=str,
        help='Optional JSON file containing mutation options.',
        default=None
    )
    parser.add_argument(
        '--n_steps', type=int,
        help='Total number of simulated annealing steps (both phases).',
        default=1000
    )
    parser.add_argument(
        '--n_init', type=int,
        help='Number of initial starting points.',
        default=8
    )
    args = parser.parse_args()

<<<<<<< HEAD
    main(args.pdb_file, args.chain, n_steps=args.n_steps, mutation_json_path=args.mutation_json, n_init=args.n_init)
=======
    main(args.pdb_file, args.chain, n_steps=args.n_steps, mutation_json_path=args.mutation_json)
>>>>>>> 4edbf1c460dad3cbe95306d5065142517646bad1
