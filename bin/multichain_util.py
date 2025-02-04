import biotite.structure
import numpy as np
import torch
from typing import Sequence, Tuple, List
from util import (
    load_structure,
    extract_coords_from_structure,
    load_coords,
    get_sequence_loss,
    get_encoder_output,
)
import util


def extract_coords_from_complex(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: biotite AtomArray
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
          coordinates representing the backbone of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    coords = {}
    seqs = {}
    all_chains = biotite.structure.get_chains(structure)
    for chain_id in all_chains:
        chain = structure[structure.chain_id == chain_id]
        coords[chain_id], seqs[chain_id] = extract_coords_from_structure(chain)
    return coords, seqs


def load_complex_coords(fpath, chains):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chains: the chain ids (the order matters for autoregressive model)
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
          coordinates representing the backbone of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    structure = load_structure(fpath, chains)
    return extract_coords_from_complex(structure)

#*
def _concatenate_coords(
        coords,
        target_chain_id,
        padding_length=10,
        order=None
):
    """
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq)
            - coords_concatenated is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between 
              AND target chain placed first
            - seq is the extracted sequence, with padding tokens inserted
            between the concatenated chains
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    if order is None:
        order = (
            [ target_chain_id ] +
            [ chain_id for chain_id in coords if chain_id != target_chain_id ]
        )
    coords_list, coords_chains = [], []
    for idx, chain_id in enumerate(order):
        if idx > 0:
            coords_list.append(pad_coords)
            coords_chains.append([ 'pad' ] * padding_length)
        coords_list.append(list(coords[chain_id]))
        coords_chains.append([ chain_id ] * coords[chain_id].shape[0])
    coords_concatenated = np.concatenate(coords_list, axis=0)
    coords_chains = np.concatenate(coords_chains, axis=0).ravel()
    return coords_concatenated, coords_chains

#*
def _concatenate_seqs(
        native_seqs,
        target_seq,
        target_chain_id,
        padding_length=10,
        order=None,
):
    """
    Args:
        native_seqs: Dictionary mapping chain ids to corresponding AA sequence
        target_seq: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        native_seqs_concatenated: Array of length L, concatenation of the chain 
        sequences with padding in between
    """
    if order is None:
        order = (
            [ target_chain_id ] +
            [ chain_id for chain_id in native_seqs if chain_id != target_chain_id ]
        )
    native_seqs_list = []
    for idx, chain_id in enumerate(order):
        if idx > 0:
            native_seqs_list.append(['<mask>'] * (padding_length - 1) + ['<cath>'])
        if chain_id == target_chain_id:
            native_seqs_list.append(list(target_seq))
        else:
            native_seqs_list.append(list(native_seqs[chain_id]))
    native_seqs_concatenated = ''.join(np.concatenate(native_seqs_list, axis=0))
    return native_seqs_concatenated


#*
def sample_sequence_in_complex(model, coords, target_chain_id, temperature=1.,
        padding_length=10):
    """
    Samples sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: padding length in between chains
    Returns:
        Sampled sequence for the target chain
    """
    target_chain_len = coords[target_chain_id].shape[0]
    all_coords, coords_chains = _concatenate_coords(coords, target_chain_id)
    device = next(model.parameters()).device

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = ['<pad>'] * all_coords.shape[0]
    for i in range(target_chain_len):
        padding_pattern[i] = '<mask>'
    sampled = model.sample(all_coords, partial_seq=padding_pattern,
            temperature=temperature, device=device)
    sampled = sampled[:target_chain_len]
    return sampled


#*
def score_sequence_in_complex(
    model,
    alphabet,
    coords,
    native_seqs,
    target_chain_id,
    target_seq_list,
    batch_converter,
    device,
    order=None
):
    """
    For each sequence in target_seq_list, merges coordinates + seq for the entire complex,
    then does a single forward pass on the batch.
    Returns (ll_fullseq_list, ll_targetseq_list) each of shape [batch_size].
    """
    import numpy as np

    all_coords_list = []
    all_seq_list = []
    coords_chains_list = []

    for seq in target_seq_list:
        merged_coords, merged_chains = _concatenate_coords(
            coords, target_chain_id, order=order
        )
        merged_seq = _concatenate_seqs(
            native_seqs, seq, target_chain_id, order=order
        )
        all_coords_list.append(merged_coords)
        all_seq_list.append(merged_seq)
        coords_chains_list.append(merged_chains)

    loss_array, pad_mask_array = get_sequence_loss(
        model, all_coords_list, all_seq_list, batch_converter, device, alphabet
    )
    # shape: [batch_size, L]

    ll_fullseq_list = []
    ll_targetseq_list = []

    for i in range(len(all_seq_list)):
        # Identify chain positions
        coords_chains = coords_chains_list[i]  # e.g. shape [L]
        loss_i = loss_array[i]
        # Negative average log-likelihood
        ll_full = -np.mean(loss_i[coords_chains != 'pad'])
        ll_target = -np.mean(loss_i[coords_chains == target_chain_id])
        ll_fullseq_list.append(ll_full)
        ll_targetseq_list.append(ll_target)

    return ll_fullseq_list, ll_targetseq_list


def get_encoder_output_for_complex(model, alphabet, coords, target_chain_id):
    """
    Args:
        model: An instance of the GVPTransformer model
        alphabet: Alphabet for the model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
    Returns:
        Dictionary mapping chain id to encoder output for each chain
    """
    all_coords = _concatenate_coords(coords, target_chain_id)
    all_rep = get_encoder_output(model, alphabet, all_coords)
    target_chain_len = coords[target_chain_id].shape[0]
    return all_rep[:target_chain_len]