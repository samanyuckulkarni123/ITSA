from Bio.PDB import PDBParser, DSSP, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import numpy as np
import pandas as pd
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# ======================== CONFIG ========================

interaction_letters = {
    "hydrogen_bonds": "H",
    "disulfide_bridge": "D",
    "ionic_bonds": "I",
    "van_der_waals": "V",
    "pi_pi_stacking": "P",
    "hydrophobic_interaction": "Y"
}

interaction_thresholds = {
    "candidate_residue_search": 12.0,
    "hydrogen_bonds": 3.5,
    "disulfide_bridge": 2.3,
    "ionic_bonds": 4.0,
    "van_der_waals": 4.5,
    "pi_pi_stacking": 5.5,
    "hydrophobic_interaction": 4.8
}

rotational_thresholds = {
    "pi_pi_stacking": 35.0
}

hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP", "GLY"}
aromatic_residues = {"PHE", "TYR", "TRP", "HIS"}
positive_residues = {"LYS", "ARG", "HIS"}
negative_residues = {"ASP", "GLU"}

HBOND_ATOMS = {
    "SER": {"OG"},
    "THR": {"OG1"},
    "ASN": {"OD1", "ND2"},
    "GLN": {"OE1", "NE2"},
    "TYR": {"OH"},
    "CYS": {"SG"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

IONIC_ATOMS = {
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}

HYDROPHOBIC_ATOMS = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"},
    "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "GLY": {"CA"},
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


# ======================== DATA CLASS ========================

@dataclass
class ResidueToken:
    interactions: List[str]
    sec_struct: str

    def primary(self) -> str:
        return self.interactions[0] if self.interactions else "U"


# ======================== HELPERS ========================

def get_atom_element(atom):
    element = getattr(atom, "element", "") or ""
    element = element.strip().upper()
    if element:
        return element
    name = atom.get_name().strip()
    return name[0].upper() if name else ""


def has_required_atoms(residue):
    return all(atom in residue for atom in ["N", "CA", "C"])


def get_heavy_atoms(residue, atom_names=None, include_backbone=True):
    atoms = []
    allowed = set(atom_names) if atom_names is not None else None

    for atom in residue:
        atom_name = atom.get_name()
        if allowed is not None and atom_name not in allowed:
            continue
        if not include_backbone and atom_name in BACKBONE_ATOMS:
            continue
        if get_atom_element(atom) != "H":
            atoms.append(atom)

    return atoms


def min_atom_distance(
    res1, res2,
    atom_names1=None, atom_names2=None,
    include_backbone1=True, include_backbone2=True
):
    atoms1 = get_heavy_atoms(res1, atom_names=atom_names1, include_backbone=include_backbone1)
    atoms2 = get_heavy_atoms(res2, atom_names=atom_names2, include_backbone=include_backbone2)

    if not atoms1 or not atoms2:
        return np.inf

    min_dist = np.inf
    for atom1 in atoms1:
        for atom2 in atoms2:
            dist = atom1 - atom2
            if dist < min_dist:
                min_dist = dist

    return float(min_dist)


def get_aromatic_centroid(residue):
    atom_names = AROMATIC_RING_ATOMS.get(residue.get_resname(), [])
    atoms = [residue[name] for name in atom_names if name in residue]
    if len(atoms) < 3:
        return None
    coords = np.array([atom.get_coord() for atom in atoms], dtype=float)
    return coords.mean(axis=0)


def get_ring_plane_normal(residue):
    atom_names = AROMATIC_RING_ATOMS.get(residue.get_resname(), [])
    atoms = [residue[name] for name in atom_names if name in residue]
    if len(atoms) < 3:
        return None

    p1, p2, p3 = (np.array(atoms[i].get_coord(), dtype=float) for i in range(3))
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    return normal / norm


def angle_between_normals(n1, n2):
    if n1 is None or n2 is None:
        return 180.0
    dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return np.degrees(np.arccos(abs(dot)))


def has_hbond_capability(residue):
    return residue.get_resname() in HBOND_ATOMS


def is_opposite_charge_pair(res1, res2):
    r1 = res1.get_resname()
    r2 = res2.get_resname()
    return (
        (r1 in positive_residues and r2 in negative_residues) or
        (r1 in negative_residues and r2 in positive_residues)
    )


def classify_interaction(res1, res2):
    res1_name = res1.get_resname()
    res2_name = res2.get_resname()

    if res1_name == "CYS" and res2_name == "CYS" and "SG" in res1 and "SG" in res2:
        sg_distance = res1["SG"] - res2["SG"]
        if sg_distance <= interaction_thresholds["disulfide_bridge"]:
            return "disulfide_bridge", sg_distance

    if is_opposite_charge_pair(res1, res2):
        ionic_distance = min_atom_distance(
            res1, res2,
            atom_names1=IONIC_ATOMS.get(res1_name, set()),
            atom_names2=IONIC_ATOMS.get(res2_name, set())
        )
        if ionic_distance <= interaction_thresholds["ionic_bonds"]:
            return "ionic_bonds", ionic_distance

    if has_hbond_capability(res1) and has_hbond_capability(res2):
        hbond_distance = min_atom_distance(
            res1, res2,
            atom_names1=HBOND_ATOMS.get(res1_name, set()),
            atom_names2=HBOND_ATOMS.get(res2_name, set())
        )
        if hbond_distance <= interaction_thresholds["hydrogen_bonds"]:
            return "hydrogen_bonds", hbond_distance

    if res1_name in aromatic_residues and res2_name in aromatic_residues:
        centroid1 = get_aromatic_centroid(res1)
        centroid2 = get_aromatic_centroid(res2)
        if centroid1 is not None and centroid2 is not None:
            centroid_distance = float(np.linalg.norm(centroid1 - centroid2))
            ring_angle = angle_between_normals(
                get_ring_plane_normal(res1),
                get_ring_plane_normal(res2)
            )
            if (
                centroid_distance <= interaction_thresholds["pi_pi_stacking"] and
                ring_angle <= rotational_thresholds["pi_pi_stacking"]
            ):
                return "pi_pi_stacking", centroid_distance

    if res1_name in hydrophobic_residues and res2_name in hydrophobic_residues:
        hydrophobic_distance = min_atom_distance(
            res1, res2,
            atom_names1=HYDROPHOBIC_ATOMS.get(res1_name, set()),
            atom_names2=HYDROPHOBIC_ATOMS.get(res2_name, set()),
            include_backbone1=False,
            include_backbone2=False
        )
        if hydrophobic_distance <= interaction_thresholds["hydrophobic_interaction"]:
            return "hydrophobic_interaction", hydrophobic_distance

    vdw_distance = min_atom_distance(
        res1, res2,
        include_backbone1=False,
        include_backbone2=False
    )
    if np.isinf(vdw_distance):
        vdw_distance = min_atom_distance(res1, res2)

    if vdw_distance <= interaction_thresholds["van_der_waals"]:
        return "van_der_waals", vdw_distance

    return None, np.inf


def assign_secondary_structure(structure, pdb_file):
    ss_map = {}

    try:
        model = next(structure.get_models())
        dssp = DSSP(model, pdb_file, file_type="PDB")

        for key in dssp.keys():
            dssp_tuple = dssp[key]
            ss_char = dssp_tuple[2] if len(dssp_tuple) >= 3 else "C"

            if ss_char in ("H", "G", "I"):
                ss = "H"
            elif ss_char in ("E", "B"):
                ss = "E"
            else:
                ss = "C"

            chain_id, res_id = key
            for m in structure:
                if chain_id not in m:
                    continue
                chain = m[chain_id]
                for res in chain:
                    if (
                        res.get_id() == res_id and
                        is_aa(res, standard=True) and
                        has_required_atoms(res)
                    ):
                        ss_map[id(res)] = ss

    except Exception:
        for model in structure:
            for chain in model:
                for res in chain:
                    if is_aa(res, standard=True) and has_required_atoms(res):
                        ss_map[id(res)] = "C"

    return ss_map


def extract_interaction_tokens(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    ss_map = assign_secondary_structure(structure, file_path)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and has_required_atoms(residue):
                    residues.append(residue)

    if not residues:
        return []

    atoms = [atom for residue in residues for atom in residue]
    neighbor_search = NeighborSearch(atoms)
    tokens: List[ResidueToken] = []

    for res1 in residues:
        ca1 = res1["CA"]
        nearby_atoms = neighbor_search.search(
            ca1.get_coord(),
            interaction_thresholds["candidate_residue_search"]
        )

        nearby_residues = {
            atom.get_parent() for atom in nearby_atoms
            if atom.get_parent() is not res1
        }

        candidate_interactions = []
        for res2 in nearby_residues:
            if not (is_aa(res2, standard=True) and has_required_atoms(res2)):
                continue

            interaction_type, interaction_distance = classify_interaction(res1, res2)
            if interaction_type:
                candidate_interactions.append((interaction_type, interaction_distance))

        candidate_interactions.sort(key=lambda x: x[1])

        seen_types = set()
        top_types = []
        for itype, _dist in candidate_interactions:
            if itype not in seen_types:
                seen_types.add(itype)
                top_types.append(itype)
            if len(top_types) >= 3:
                break

        letters = ["U"] if not top_types else [interaction_letters[t] for t in top_types]
        ss = ss_map.get(id(res1), "C")
        tokens.append(ResidueToken(interactions=letters, sec_struct=ss))

    return tokens


def serialize_tokens(tokens: List[ResidueToken]) -> str:
    return json.dumps([
        {
            "interactions": t.interactions,
            "sec_struct": t.sec_struct
        }
        for t in tokens
    ])


def build_pdb_path(base_dir: str, family: str, pdb_id: str) -> str:
    return os.path.join(base_dir, str(family).strip(), f"{str(pdb_id).strip().lower()}.pdb")


def process_one(row: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    pdb_id = str(row["pdb_id"]).strip()
    family = str(row["family"]).strip()
    sf_id = row.get("sf_id", "")
    prot_class = row.get("class", "")

    pdb_path = build_pdb_path(base_dir, family, pdb_id)

    out = {
        "pdb_id": pdb_id,
        "family": family,
        "sf_id": sf_id,
        "class": prot_class,
        "pdb_path": pdb_path,
        "status": "ok",
        "token_length": 0,
        "primary_seq": "",
        "tokens_json": "",
        "error": ""
    }

    if not os.path.exists(pdb_path):
        out["status"] = "missing_file"
        return out

    try:
        tokens = extract_interaction_tokens(pdb_path)
        if not tokens:
            out["status"] = "empty_tokens"
            return out

        out["token_length"] = len(tokens)
        out["primary_seq"] = "".join(t.primary() for t in tokens)
        out["tokens_json"] = serialize_tokens(tokens)

    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)

    return out


def main():
    parser = argparse.ArgumentParser(description="Tokenize all proteins from master_dataset.csv once.")
    parser.add_argument("--input_csv", default="/mnt/data/master_dataset.csv", help="Path to master dataset CSV")
    parser.add_argument("--base_dir", default="SCOP_Dataset", help="Base directory containing family/pdbid.pdb files")
    parser.add_argument("--output_csv", default="protein_tokens.csv", help="Output CSV for token cache")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = {"pdb_id", "family"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Deduplicate just in case
    dedup_cols = [c for c in ["pdb_id", "family", "sf_id", "class"] if c in df.columns]
    df = df.drop_duplicates(subset=["pdb_id", "family"]).reset_index(drop=True)

    rows = df.to_dict(orient="records")
    results = []

    if args.workers <= 1:
        for i, row in enumerate(rows, start=1):
            results.append(process_one(row, args.base_dir))
            if i % 10 == 0 or i == len(rows):
                print(f"Processed {i}/{len(rows)}")
    else:
        max_workers = min(args.workers, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, row, args.base_dir) for row in rows]
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % 10 == 0 or completed == len(rows):
                    print(f"Processed {completed}/{len(rows)}")

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(["family", "pdb_id"]).reset_index(drop=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"\nSaved token cache to {args.output_csv}")
    print("\nStatus counts:")
    print(out_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()