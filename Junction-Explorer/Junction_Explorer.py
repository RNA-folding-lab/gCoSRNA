from collections import deque, defaultdict
from draw_rna.ipynb_draw import draw_struct
from math import log
import math 
import pandas as pd
import joblib
import os
# ---------------------------
# Free energy of base-stacking from Turner's parameters
# ---------------------------
STACKING_ENERGY = {
    'AUAU': -0.93, 'UAUA': -0.93, 'AUUA': -1.10, 'UAAU': -1.33,
    'CGUA': -2.08, 'AUGC': -2.08, 'CGAU': -2.11, 'UAGC': -2.11,
    'GCUA': -2.24, 'AUCG': -2.24, 'GCAU': -2.35, 'UACG': -2.35,
    'CGGC': -2.36, 'GCGC': -3.26, 'CGCG': -3.26, 'GCCG': -3.42,
    'AUGU': -0.55, 'UGUA': -0.55, 'AUUG': -1.36, 'GUUA': -1.36,
    'CGGU': -1.41, 'UGGC': -1.41, 'CGUG': -2.11, 'GUGC': -2.11,
    'GCGU': -1.53, 'UGCG': -1.53, 'GCUG': -2.51, 'GUCG': -2.51,
    'GUAU': -1.27, 'UAUG': -1.27, 'GUGU':  0.47, 'UGUG':  0.47,
    'GUUG':  1.29, 'UAGU': -1.0, 'UGAU': -1.0, 'UGGU':  0.3
}
INITIATION_ENERGY = 4.09
AU_END_PENALTY = 0.45
GU_END_PENALTY = 0.45
SYMMETRY_CORRECTION = 0.43
watson_crick = ['AU', 'UA', 'CG', 'GC']
mismatch_para = {
    'AU': {'A': {'A': -0.8, 'C': -1.0, 'G': -0.8, 'U': -1.0},
           'C': {'A': -0.6, 'C': -0.7, 'G': -0.6, 'U': -0.7},
           'G': {'A': -0.8, 'C': -1.0, 'G': -0.8, 'U': -1.0},
           'U': {'A': -0.6, 'C': -0.8, 'G': -0.6, 'U': -0.8}},
    'CG': {'A': {'A': -1.5, 'C': -1.5, 'G': -1.4, 'U': -1.5},
           'C': {'A': -1.0, 'C': -1.1, 'G': -1.0, 'U': -0.8},
           'G': {'A': -1.4, 'C': -1.5, 'G': -1.6, 'U': -1.5},
           'U': {'A': -1.0, 'C': -1.4, 'G': -1.0, 'U': -1.2}},
    'GC': {'A': {'A': -1.1, 'C': -1.5, 'G': -1.3, 'U': -1.5},
           'C': {'A': -1.1, 'C': -0.7, 'G': -1.1, 'U': -0.5},
           'G': {'A': -1.6, 'C': -1.5, 'G': -1.4, 'U': -1.5},
           'U': {'A': -1.1, 'C': -1.0, 'G': -1.1, 'U': -0.7}},
    'GU': {'A': {'A': -0.3, 'C': -1.0, 'G': -0.8, 'U': -1.0},
           'C': {'A': -0.6, 'C': -0.7, 'G': -0.6, 'U': -0.7},
           'G': {'A': -0.6, 'C': -1.0, 'G': -0.8, 'U': -1.0},
           'U': {'A': -0.6, 'C': -0.8, 'G': -0.6, 'U': -0.8}},
    'UA': {'A': {'A': -1.0, 'C': -0.8, 'G': -1.1, 'U': -0.8},
           'C': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5},
           'G': {'A': -1.1, 'C': -0.8, 'G': -1.2, 'U': -0.8},
           'U': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5}},
    'UG': {'A': {'A': -1.0, 'C': -0.8, 'G': -1.1, 'U': -0.8},
           'C': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5},
           'G': {'A': -0.5, 'C': -0.8, 'G': -0.8, 'U': -0.8},
           'U': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5}},
}
# ---------------------------
# Utility Functions
# ---------------------------

def get_sequence_from_indices(sequence, indices):
    """Return a string of bases from given indices."""
    return ''.join(sequence[i] for i in indices)

def get_continuous_unpaired_regions(start, end, exclude_set):
    """Return list of unpaired base indices between start and end."""
    return [i for i in range(start + 1, end) if i not in exclude_set]

# ---------------------------
# Dot-bracket Parsing
# ---------------------------

def parse_dot_bracket(dot_bracket):
    """
    Parse dot-bracket notation and return a dictionary of valid base pairs.
    Isolated base pairs (not part of stems) are excluded.
    """
    stack = []
    raw_pairs = {}

    # Initial base pair mapping
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                raw_pairs[i] = j
                raw_pairs[j] = i

    # Filter out isolated base pairs
    valid = set()
    for i in sorted(raw_pairs):
        j = raw_pairs[i]
        if i < j:
            if ((i + 1 in raw_pairs and raw_pairs[i + 1] == j - 1) or
                (i - 1 in raw_pairs and raw_pairs[i - 1] == j + 1)):
                valid.add(i)
                valid.add(j)

    return {i: raw_pairs[i] for i in valid}

# ---------------------------
# Stem Extraction
# ---------------------------

def extract_stems(pairs):
    """
    Extract stems as contiguous base pair segments.
    """
    visited = set()
    stems = []

    for i in sorted(pairs):
        j = pairs[i]
        if i < j and i not in visited:
            stem = []
            while i in pairs and pairs[i] == j:
                stem.append((i, j))
                visited.add(i)
                visited.add(j)
                i += 1
                j -= 1
            stems.append(stem)

    return stems

# ---------------------------
# multibranch Detection
# ---------------------------

def detect_hairpin(stem, sequence):
    """
    Identify hairpin loop closed by a given stem.
    """
    i, j = stem[-1]
    return {
        "range": (i + 1, j - 1),
        "sequence": sequence[i + 1:j]
    }

def detect_internal_or_bulge_or_multibranch(stem, internal_stems, sequence, paired_indices):
    """
    Identify internal or multibranch loops between stems.
    """
    i, j = stem[-1]
    loop_info = {"internal": [], "multibranch": [],"bulge":[]}
    branches = []
    start = i

    for inner_start, inner_end in internal_stems:
        region = get_continuous_unpaired_regions(start, inner_start, paired_indices)
        branches.append((start, inner_start, region))
        start = inner_end

    # Final branch
    last_region = get_continuous_unpaired_regions(start, j, paired_indices)
    branches.append((start, j, last_region))

    unpaired_branches = [b for b in branches if b[2]]
    branch_count = len([1 for _, _, r in branches if r])

    if branch_count > 2:
        loop_info["multibranch"].append({
            "stem": (i, j),
            "branches": [(r[0], r[-1], get_sequence_from_indices(sequence, r)) if r else (e, e, '') 
                         for (_, e, r) in branches],
            "branch_count": len(branches)
        })
    elif branch_count == 2:
        loop_info["internal"].append({
            "stem": (i, j),
            "branches": [(r[0], r[-1], get_sequence_from_indices(sequence, r)) if r else (e, e, '') 
                         for (_, e, r) in branches],
            "branch_count": len(branches)
        })
    else:
        for _, _, r in branches:
            if r:
                loop_info["bulge"].append({
                    "branches": [(r[0], r[-1], get_sequence_from_indices(sequence, r))],
                    "branch_count": 2,  
                    "5_range": (r[0], r[-1]),
                    "5_seq": get_sequence_from_indices(sequence, r),
                    "3_range": None,
                    "3_seq": '',
                    "stem": (i, j)
                })

    return loop_info

def detect_dangling_ends(sequence, paired_indices, loop_indices):
    """
    Identify unpaired nucleotides at the 5' and 3' ends.
    """
    total_len = len(sequence)
    all_indices = set(range(total_len))
    unpaired = sorted(i for i in all_indices if i not in paired_indices and i not in loop_indices)

    d5, d3 = [], []
    
    # Detect 5' dangling end
    i = 0
    while i < total_len and i in unpaired:
        i += 1
    if i > 0:
        d5.append({"range": (0, i - 1), "sequence": sequence[0:i]})

    # Detect 3' dangling end
    j = total_len - 1
    while j >= 0 and j in unpaired:
        j -= 1
    if j < total_len - 1:
        d3.append({"range": (j + 1, total_len - 1), "sequence": sequence[j + 1:]})

    return d5, d3

# ---------------------------
# Main Structure Classification
# ---------------------------

def identify_loops(sequence, dot_bracket, stems):
    """
    Identify various RNA secondary structure elements:
    - Hairpin loops
    - Internal loops
    - Multibranch loops
    - Dangling ends (5' and 3')
    """
    pairs = parse_dot_bracket(dot_bracket)
    paired_indices = set(pairs.keys())
    stem_ranges = sorted((s[0][0], s[0][1]) for s in stems)

    loop_info = {
        "hairpin": [],
        "bulge":[],
        "internal": [],
        "multibranch": [],
        "dangling_5prime": [],
        "dangling_3prime": []
    }

    # Detect hairpins, internal loops, bulge loops, and multibranch loops
    for stem in stems:
        i, j = stem[-1]
        internal_stems = [s for s in stem_ranges if i < s[0] < s[1] < j]
        if not internal_stems:
            loop_info["hairpin"].append(detect_hairpin(stem, sequence))
        else:
            res = detect_internal_or_bulge_or_multibranch(stem, internal_stems, sequence, paired_indices)
            loop_info["bulge"].extend(res["bulge"])
            loop_info["internal"].extend(res["internal"])
            loop_info["multibranch"].extend(res["multibranch"])

    # Collect loop indices
    loop_indices = set()
    for ltype in ["hairpin", "bulge","internal", "multibranch"]:
        for entry in loop_info[ltype]:
            if ltype == "multibranch":
                for r in entry["branches"]:
                    loop_indices.update(range(r[0], r[1] + 1))
            elif ltype == "internal":
                for r in entry["branches"]:
                    loop_indices.update(range(r[0], r[1] + 1))
            elif ltype == "bulge":
                if entry["5_range"]:
                    loop_indices.update(range(entry["5_range"][0], entry["5_range"][1] + 1))
                if entry["3_range"]:
                    loop_indices.update(range(entry["3_range"][0], entry["3_range"][1] + 1))
            else:
                loop_indices.update(range(entry["range"][0], entry["range"][1] + 1))

    # Detect dangling ends
    d5, d3 = detect_dangling_ends(sequence, paired_indices, loop_indices)
    loop_info["dangling_5prime"].extend(d5)
    loop_info["dangling_3prime"].extend(d3)

    return loop_info


# Build a mapping from base index to stem ID
def build_base_to_stem_map(stems):
    base_to_stem = {}
    for stem_id, stem in enumerate(stems):
        for i, j in stem:
            base_to_stem[i] = stem_id
            base_to_stem[j] = stem_id
    return base_to_stem

# Extract sorted unique indices and corresponding subsequence and dot-bracket
def extract_subsequence_and_structure(sequence, dot_bracket, indices):
    sorted_indices = sorted(set(i for i in indices if 0 <= i < len(sequence)))
    sub_seq = ''.join(sequence[i] for i in sorted_indices)
    sub_db = ''.join(dot_bracket[i] for i in sorted_indices)
    return sub_seq, sub_db, sorted_indices

def extract_substructures(sequence, dot_bracket, stems, loop_info, ltype):
    substructures = []
    base_to_stem = build_base_to_stem_map(stems)

    for loop_id, loop in enumerate(loop_info[ltype]):
        branch_indices = set()
        involved_stem_ids = set()

        # Collect branch indices and neighboring stem IDs
        for left, right, _ in loop["branches"]:
            branch_indices.update(range(left, right + 1))
            if left - 1 in base_to_stem:
                involved_stem_ids.add(base_to_stem[left - 1])
            if right + 1 in base_to_stem:
                involved_stem_ids.add(base_to_stem[right + 1])

        # Collect all indices from involved stems
        stem_indices = set()
        for sid in involved_stem_ids:
            for i, j in stems[sid]:
                stem_indices.update([i, j])

        all_indices = branch_indices.union(stem_indices)
        sub_seq, sub_db, sorted_indices = extract_subsequence_and_structure(sequence, dot_bracket, all_indices)

        substructures.append({
            "loop_id": loop_id + 1,
            "sequence": sub_seq,
            "dot_bracket": sub_db,
            "stems": [stems[sid] for sid in involved_stem_ids],
            "loop_info": loop,
            "original_indices": sorted_indices
        })

    return substructures



def get_longest_consecutive_A(seq):
    """get the longest consecutive A in loop"""
    max_run = run = 0
    for base in seq:
        if base == 'A':
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

## Calculate the stem free energy
def is_self_complementary(chain1, chain2):
    """
    Check if two nucleotide chains are reverse complements of each other.
    Used to determine symmetry for energy correction.
    Returns:
        bool: True if self-complementary, False otherwise.
    """
    base_pair_set = {'AU', 'UA', 'GC', 'CG'}
    n = len(chain1)
    return all(chain1[i] + chain2[n - 1 - i] in base_pair_set for i in range(n))

def calculate_stem_free_energy(stem, sequence):
    """
    Calculate the free energy of an RNA stem based on base pair stacking,
    terminal mismatches, and self-complementary corrections.
    Returns:
        float: Total stem free energy (in kcal/mol).
    """
    if not stem:
        return 0.0

    chain1 = ''.join(sequence[i] for i, _ in stem)
    chain2 = ''.join(sequence[j] for _, j in stem)

    energy = INITIATION_ENERGY

    # Terminal mismatch penalty (only first and last base pair)
    if chain1[0] + chain2[0] in ['AU', 'UA']:
        energy += AU_END_PENALTY
    elif chain1[0] + chain2[0] in ['GU', 'UG']:
        energy += GU_END_PENALTY
    if chain1[-1] + chain2[-1] in ['AU', 'UA']:
        energy += AU_END_PENALTY
    elif chain1[-1] + chain2[-1] in ['GU', 'UG']:
        energy += GU_END_PENALTY

    # Stacking energies between consecutive base pairs
    for k in range(len(chain1) - 1):
        key = chain1[k] + chain2[k] + chain1[k+1] + chain2[k+1]      # Stack order may be ERROR
        energy += STACKING_ENERGY.get(key, 0.0)

    if is_self_complementary(chain1, chain2): # Symmetry correction for self-complementary stems
        energy += SYMMETRY_CORRECTION

    return round(energy, 2)

def calculate_G(Li, terminal_pairs):
    """
    Estimate the free energy contribution of coaxial stacking at a loop junction.
    Returns:
        float: Estimated coaxial stacking free energy (in kcal/mol).
    """
    a = 9.3
    b= 0
    c = -0.6

    if terminal_pairs in STACKING_ENERGY:
        G0 = STACKING_ENERGY[terminal_pairs]
    elif terminal_pairs[0:2] in mismatch_para:
        G0 = mismatch_para[terminal_pairs[0:2]][terminal_pairs[2]][terminal_pairs[3]]
    else:
        G0 = 0.0
    if Li <= 1:
        G = G0   
    elif 2 <= Li <= 6:
        G = a + b * Li + 2*c
    else:
        G = a + 6*b + 1.1 * math.log(Li / 6) + 2*c
    return round(G, 2)

def compute_coaxial_stacking_energy(junction_bases, loop_length):
    """
    Compute the coaxial stacking free energy based on junction interface bases and loop length.
    Returns:
        float: Coaxial stacking free energy (in kcal/mol).
    """
    #print("~~~~~",junction_bases)
    if not junction_bases or len(junction_bases) != 4:
        return 0.0

    try:
        return calculate_G(loop_length, junction_bases)
    except Exception as e:
        print("Error in compute_coaxial_stacking_energy:", e)
        return 0.0

def find_terminal_pair_connected_to_loop(stem, loop_start, loop_end):
    """
    Identify which terminal base pair of a stem is adjacent to a given loop.
    Returns:
        tuple: Terminal base pair (i, j) closest to the loop, or None if not found.
    """
    candidates = [stem[0], stem[-1]]  # Only consider terminal pairs
    for i, j in candidates:
        print(i,j)
        if loop_start - 1 in (i, j) or loop_end + 1 in (i, j) or loop_start in (i,j) or loop_end in (i,j):
            if loop_start - 1 == i or loop_end + 1 == i or loop_start == i or loop_end == i:
                return (i, j)  # i is loop-adjacent
            else:
                return (j, i)  # j is loop-adjacent
    return None


def compute_multibranch_features(multibranch_sub, total_branch_count, full_stem_list,sequence):
    # Loop features
    loop_lengths = [len(loop[2]) for loop in multibranch_sub['loop_info']['branches']]
    loop_A_counts = [get_longest_consecutive_A(loop[2]) for loop in multibranch_sub['loop_info']['branches']]
    loop_order = sorted(loop_lengths)

    # Coaxial stacking free energy between two stems
    coaxial_dG_list = []
    for id in range(0,total_branch_count):
        id_2 = (id + 1) % total_branch_count
        stem_ids = [id,id_2]
        coaxial_dG = 0.0
        if len(stem_ids) == 2:
            stem1 = full_stem_list[stem_ids[0]]
            stem2 = full_stem_list[stem_ids[1]]
            between_loops = [multibranch_sub['loop_info']['branches'][id]]
            order = [id-1,id,id_2]
            loops = [multibranch_sub['loop_info']['branches'][i] for i in order]
            # print("between_loops",between_loops)
            if len(between_loops) == 1:
                loop_length = len(between_loops[0][2])
            elif len(between_loops) == 2:
                loop_length = abs(len(between_loops[0][2]) - len(between_loops[1][2])) ## For internal
            else:
                loop_length = 0
            
            if between_loops:
                loop_start, loop_end, _ = between_loops[0]

                stem1_pair = find_terminal_pair_connected_to_loop(stem1, loop_start, loop_end)
                stem2_pair = find_terminal_pair_connected_to_loop(stem2, loop_start, loop_end)


                if stem1_pair and stem2_pair:
                    junction_bases = ""
                    A, B = stem1_pair
                    C, D = stem2_pair
                    if loop_length == 1:
                        if loops[0][2] and loops[2][2] == '':
                            junction_bases = sequence[A] + sequence[B] + between_loops[0][2] + loops[0][2][-1]
                            coaxial_dG = compute_coaxial_stacking_energy(junction_bases, loop_length)
                        elif loops[2][2] and loops[0][2] == '':
                            junction_bases = sequence[D] + sequence[C] + loops[2][2][0] + between_loops[0][2]
                            coaxial_dG = compute_coaxial_stacking_energy(junction_bases, loop_length)
                        else:
                            junction_bases_1 = sequence[A] + sequence[B] + between_loops[0][2] + loops[0][2][-1]
                            junction_bases_2 = sequence[D] + sequence[C] + loops[2][2][0] + between_loops[0][2]
                            coaxial_dG_1 = compute_coaxial_stacking_energy(junction_bases_1, loop_length)
                            coaxial_dG_2 = compute_coaxial_stacking_energy(junction_bases_2, loop_length)
                            coaxial_dG = min(coaxial_dG_1,coaxial_dG_2)
                    else:
                        junction_bases = sequence[A] + sequence[B] + sequence[C] + sequence[D]
                        coaxial_dG = compute_coaxial_stacking_energy(junction_bases, loop_length)
        coaxial_dG_list.append(coaxial_dG)
        # Stem features
        
    loop_compare = {}
    if total_branch_count == 3:
        loop_compare = {
            "Min(L1,L2)": min(loop_lengths[0],loop_lengths[1]),
            "Min(L2,L3)": min(loop_lengths[1],loop_lengths[2]),
            "Min(L1,L3)": min(loop_lengths[0],loop_lengths[2])
        }
    if total_branch_count == 4:
        loop_compare = {
            "Min(L1,L3)": min(loop_lengths[0],loop_lengths[2]),
            "Min(L2,L4)": min(loop_lengths[1],loop_lengths[3])
        }
    
    return {
        "loop_lengths": loop_lengths,
        "loop_order": loop_order,
        "loop_A_counts": loop_A_counts,
        "loop_compare": loop_compare,
        "coaxial_dG": coaxial_dG_list,      
    }


def extract_multibranch_substructures_v2(sub, sequence, dot_bracket):
    """
    Extract pseudo two-way junction substructures from a multi-branched RNA region.
    Parameters:
    - sub: dictionary describing a multibranch loop, including "stems" and "loop_info".
    - sequence: full RNA sequence.
    - dot_bracket: full dot-bracket structure notation.

    Returns:
    A list of pseudo two-way junction substructures with annotated features.
    """
    stems = sub["stems"]
    branches = sub["loop_info"]["branches"]
    base_to_stem = build_base_to_stem_map(stems)
    multibranch_sub = {}
    used_pairs = set()
    total_branch_count = len(stems)

    # Compute structural and thermodynamic features
    sub["features"] = compute_multibranch_features(sub, total_branch_count, stems, sequence)
    # sub.append(multibranch_sub)
    print("====multibranch_subs====")
    print(sub)
    return sub


def extract_multibranch_features_dataframe(sub):#把sub的feature变为DataFrame格式
    """
    Extract structural features from multiple pseudo bifurcation substructures
    and return them in a pandas DataFrame format.
    """
    feature_rows = []
    print("===sub===",sub)
    
        
    features = sub["features"]

    # Get loop lengths and adenine counts (upstream, central, downstream)
    loop_lengths = features.get("loop_lengths")
    loop_As = features.get("loop_A_counts")
    loop_Compare = features.get("loop_compare")
    coaxial_dG_list = features.get("coaxial_dG")
    loop_order = features.get("loop_order")

    '''
    row = {
        "id": idx + 1, 
        "l0": loop_lengths[0],   # Length of between loop
        "l1": loop_lengths[1],   # Length of 5' loop
        "l2": loop_lengths[2],   # Length of 3' loop
        "lA1": loop_As[0],
        "lA2": loop_As[1],
        "lA3": loop_As[2],       # Number of A bases in each loop
        "G12": features.get("coaxial_dG", 0.0), # Free energy of coaxial stacking
        "G1": features.get("stem1_dG", 0.0),     # Free energy of first stem
        "G2": features.get("stem2_dG", 0.0),
        "ls1": features.get("stem_lengths", [0, 0])[0],  # Length of first stem
        "ls2": features.get("stem_lengths", [0, 0])[1],
        "internal_branch_count": features.get("internal_branch_count", 0)  # Number of internal branches
    }
    '''
    if len(sub['stems']) == 3 :
        row = {
            # "id": idx + 1,
            "|L1|": loop_lengths[0],
            "|L2|": loop_lengths[1],
            "|L3|": loop_lengths[2],
            "Min(|L2|,|L3|)":loop_Compare["Min(L2,L3)"],
            "Min(|L1|,|L3|)":loop_Compare["Min(L1,L3)"],
            "Min(|L1|,|L2|)":loop_Compare["Min(L1,L2)"],
            "|L1'|":loop_order[0],
            "|L2'|":loop_order[1],
            "|L3'|":loop_order[2],
            "G(H1,H2)": coaxial_dG_list[0],
            "G(H2,H3)": coaxial_dG_list[1],
            "G(H3,H1)": coaxial_dG_list[2],
            "A(L1)": loop_As[0],
            "A(L2)": loop_As[1],
            "A(L3)": loop_As[2]
        }
    elif len(sub['stems']) == 4:
        row = {
            # "id": idx + 1,
            "|L1|": loop_lengths[0],
            "|L2|": loop_lengths[1],
            "|L3|": loop_lengths[2],
            "|L4|": loop_lengths[3],
            "Min(|L2|,|L4|)":loop_Compare["Min(L2,L4)"],           
            "Min(|L1|,|L3|)":loop_Compare["Min(L1,L3)"],
            "|L1'|":loop_order[0],
            "|L2'|":loop_order[1],
            "|L3'|":loop_order[2],
            "|L4'|":loop_order[3],
            "G(H1,H2)": coaxial_dG_list[0],
            "G(H2,H3)": coaxial_dG_list[1],
            "G(H3,H4)": coaxial_dG_list[2],
            "G(H4,H1)": coaxial_dG_list[3],
            "A(L1)": loop_As[0],
            "A(L2)": loop_As[1],
            "A(L3)": loop_As[2],
            "A(L4)": loop_As[3]            
        }

    feature_rows.append(row)


    return pd.DataFrame(feature_rows)



def predict_coaxial_probabilities(sub, model_path):
    """
    Predict the probability of coaxial stacking between two stems in each pseudo bifurcation substructure using a pre-trained model.
    Returns a DataFrame that includes the predicted probabilities.
    """
    df = extract_multibranch_features_dataframe(sub)

     # Load the pre-trained classification model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found:{model_path}")
    model = joblib.load(model_path)

    
    coaxial_pred = model.predict(df)
    df['coaxial_pred'] = coaxial_pred
    return df

from draw_rna.ipynb_draw import draw_struct
from math import log
import math 
import pandas as pd
import joblib
import os
from pprint import pprint

# === MAIN: RNA 2D Structure Parsing & Substructure Extraction ===
def predict_coaxial_stacking(sequence, dot_bracket,visualize=False):
    ## Visualize the structure
    # draw_struct(sequence, dot_bracket)
    # === Base Pair Parsing ===
    print("Note: This parser ignores pseudoknots and isolated base pairs.")
    print("      Please ensure only standard nested base pairs use '()'.")
    print("      Pseudoknots like '[', ']', '{', '}' are ignored.\n")

    pairs = parse_dot_bracket(dot_bracket)
    stems = extract_stems(pairs)
    loop_info = identify_loops(sequence, dot_bracket, stems)


    # === Extract Multibranch Substructures ===
    multibranch_substructures = extract_substructures(sequence, dot_bracket, stems, loop_info,'multibranch')

    for sub in multibranch_substructures:
        print(f"\n=== Multibranch Substructure {sub['loop_id']} ===")
        print("Sequence:     ", sub['sequence'])
        print("Dot-bracket:  ", sub['dot_bracket'])
        print("Stems:")
        for stem in sub['stems']:
            print(" ", stem)
        print("Loop Info:    ", sub['loop_info'])

    # We only demonstrate on the first multibranch loop substructure
    if multibranch_substructures:
        sub_with_features = extract_multibranch_substructures_v2(sub,sequence,dot_bracket)
        if len(sub['stems']) == 3:
            model_path = "Schlick_three_way_model.pkl"  ##!!!change your model path
        if len(sub['stems']) == 4:
            model_path = "Schlick_four_way_model.pkl"  ##!!!change your model path
        print("model_path:",model_path)
        pred_df = predict_coaxial_probabilities(sub_with_features, model_path)
        print("pred_df\n",pred_df)
        print(f"In this multibranch_substructures: {pred_df['coaxial_pred'].iloc[0]} are predicted to be coaxially stacked.")

        


    
