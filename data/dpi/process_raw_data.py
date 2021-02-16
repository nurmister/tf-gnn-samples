import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from progressbar import progressbar
from biopandas.mol2 import split_multimol2, PandasMol2
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(1)


################################# Variables ####################################
# Map elements to integers.
atom_mapping = {
	"Br": 1,
	"C": 2,
	"Cl": 3,
	"F": 4,
	"H": 5,
	"I": 6,
	"N": 7,
	"N1+": 8,
	"O": 9,
	"O1-": 10,
	"P": 11,
	"S": 12,
	"S1-": 13,
	"Si": 14,
}

# Map ligand bonds to integers. (Need to determine types).
bond_mapping = {"1": 1, "2": 2, "3": 3, "am": 4, "ar": 5}

# Load all of the DUD-E target names.
all_targets = pd.read_csv("dud-e_targets.csv").target_name.tolist()
all_targets = [target.lower() for target in all_targets]
# Remove proteins without bond information.
all_targets.remove("drd3")
all_targets.remove("aa2ar")
all_targets.remove("thrb")

np.random.shuffle(all_targets)

all_targets = all_targets[:5]

################################# Proteins #####################################

## Text
def get_protein_text(target):
	"""Open the target's PDB file and read in the lines in a list."""
	with open(f"raw/pdb/{target}.pdb", "r") as f:
		return [entry.split() for entry in f.read().split("\n")][5:-3]


def get_protein_atoms(protein_text):
	"""Get details on target atom elements and coordinates."""
	atoms = np.array([entry[3:] for entry in protein_text if entry[0] == "ATOM"])[
		:, [2, 3, 4, 7]
	]
	atom_attribs = atoms[:, -1]
	atom_attribs = [[atom_mapping[entry]] for entry in atom_attribs.tolist()]
	return np.array(atoms[:, [0, 1, 2]]).astype(float), atom_attribs


## Bonds
def get_protein_chemical_bonds(protein_text):
	"""Get the protein's chemical bonds."""
	bonds = [entry[1:] for entry in protein_text if entry[0] == "CONECT"]
	bonds = [[int(element) - 1 for element in entry] for entry in bonds]
	processed_bonds = []
	for i in range(len(bonds)):
		for j in range(1, len(bonds[i])):
			# Add both "directions" of the bond to the list, if
			# they do not already exist in it.
			if [bonds[i][0], bonds[i][j]] not in processed_bonds:
				processed_bonds.append(
					[bonds[i][0], len(bond_mapping.keys()) + 1, bonds[i][j]]
				)
			if [bonds[i][j], bonds[i][0]] not in processed_bonds:
				processed_bonds.append(
					[bonds[i][j], len(bond_mapping.keys()) + 1, bonds[i][0]]
				)
	return processed_bonds


def get_bonds_cutoff(atom_coords, bond_cutoff):
	"""Get bond list using the proximity-based method."""
	# Create a KD tree out of the atoms - this will help
	# find an atom's nearest neighbors efficiently.
	KD_tree = cKDTree(atom_coords, leafsize=32)
	processed_bonds = []
	# For each atom, find its neighbors within the cutoff,
	# and add the associated "bonds" to processed_bonds.
	for atom_index in range(atom_coords.shape[0]):
		index_atom_atom_coords = atom_coords[atom_index, :]
		neighbor_indices = KD_tree.query_ball_point(index_atom_atom_coords, bond_cutoff)
		# Remove the index of the current atom from the
		# neighbors list, to avoid inducing self-loops.
		neighbor_indices.remove(atom_index)
		# Add both "directions" of the bond to the list.
		processed_bonds += [
			[atom_index, len(bond_mapping.keys()) + 1, neighbor_index]
			for neighbor_index in neighbor_indices
		]
		processed_bonds += [
			[neighbor_index, len(bond_mapping.keys()) + 1, atom_index]
			for neighbor_index in neighbor_indices
		]
	return processed_bonds


## Pocket
def get_ligand_centroid(target):
	"""Get the coordinates of the crystal ligand."""
	# Read in the DataFrame of the ligand, and return the
	# coordinates as a numpy array.
	ligand = PandasMol2().read_mol2(f"raw/{target}/crystal_ligand.mol2").df
	return ligand.iloc[:, 2:5].to_numpy().mean(0)


def get_pocket_indices(target, target_coords, protein_cutoff):
	"""Get the indices of target atoms that are within its pocket."""
	ligand_centroid = get_ligand_centroid(target)
	dist_from_centroid = lambda target_atom_coords: np.linalg.norm(
		target_atom_coords - ligand_centroid
	)
	target_atom_dists = np.apply_along_axis(dist_from_centroid, 1, target_coords)
	# Return the indices of the target atoms that are less than
	# the cutoff away from the centroid of the crystal ligand.
	return np.where(target_atom_dists <= protein_cutoff)[0]


def filter_bonds(bonds, pocket_indices):
	"""Get the bond list of bonds between atoms in the target's pocket."""
	# Keep only the bonds in the processed bonds list that are in the
	# target's pocket.
	filtered_bonds = [
		entry
		for entry in bonds
		if entry[0] in pocket_indices and entry[2] in pocket_indices
	]
	# Remap the entry indices to be between 0-number of atoms remaining,
	# to correspond to their positions in the coordinates array.
	index_mapping = dict(zip(pocket_indices, range(1, len(pocket_indices) + 1)))
	return [
		[index_mapping[entry[0]], entry[1], index_mapping[entry[2]]]
		for entry in filtered_bonds
	]


## Put it all together
def process_target(target, bond_mode, protein_cutoff=10, bond_cutoff=5):
	"""Get a Data object after processing the target's PDB."""
	protein_text = get_protein_text(target)
	atom_coords, atom_attribs = get_protein_atoms(protein_text)
	# Get the bond list according to the bond_mode.
	if bond_mode == "chemical":
		bonds = get_chemical_bonds(protein_text)
	elif bond_mode == "bond_cutoff":
		bonds = get_bonds_cutoff(atom_coords, bond_cutoff)
	else:
		raise Exception("Invalid bond mode.")
	pocket_indices = get_pocket_indices(target, atom_coords, protein_cutoff)
	# Filter out all non-pocket atom information from
	# atom_attribs and atom_coords.
	atom_attribs = (np.array(atom_attribs)[pocket_indices, :]).tolist()
	bonds = filter_bonds(bonds, pocket_indices)
	return {
		"graph": bonds,
		"node_features": atom_attribs,
	}


################################## Ligands #####################################
def process_ligand_text(text, num_atoms_target):
	"""Get atom attributes and bond information for the current ligand."""
	num_atoms = int(text[2].split()[0])
	# Remove extraneous information from each line of the text.
	cleaned_text = [row[:-1] for row in text[7:]]
	# Split the cleaned text into two.
	atom_attribs, bonds = (
		[row.split()[5].split(".")[0] for row in cleaned_text[:num_atoms]],
		[row.split() for row in cleaned_text[(num_atoms + 1) :]],
	)
	atom_attribs = [[atom_mapping[element]] for element in atom_attribs]
	# Process bond information into integers.
	bonds = [
		[int(bond[1]) + num_atoms_target, bond_mapping[bond[3]], int(bond[2]) + num_atoms_target]
		for bond in bonds
	]
	# Create a Data object out of the ligand information, and add it to
	# the ligand dictionary for this target.
	return [atom_attribs, bonds]


def process_ligands(target):
	"""Get information for all ligands associated with the target."""
	ligand_list = []
	num_atoms_target = len(protein_dict[target]["node_features"])
	for fname in ["actives_final.mol2", "decoys_final.mol2"]:
		response = int(fname.startswith("a"))
		# Split the mol2 file with multiple ligands by ligand.
		# This list will be a list of pair sub-lists, the first
		# element of which is the ligand code, and the second of which
		# is the associated coordinate and bond text.
		curr_info = list(split_multimol2(f"raw/{target}/{fname}"))
		curr_info = [
			[f"{target}_{entry[0]}"] + process_ligand_text(entry[1], num_atoms_target) + [[[response]]]
			for entry in curr_info
		]
		curr_info = [
			dict(zip(("id", "node_features", "graph", "targets"), curr_info[i]))
			for i in range(len(curr_info))
		]
		ligand_list += curr_info
	return ligand_list


################################ Application ###################################
if not os.path.isfile("ligand_dict.pkl") or not os.path.isfile("combined_list.pkl"):
	if not os.path.isfile("protein_dict.pkl"):
		print("Creating protein dictionary")
		protein_dict = {}
		for target in progressbar(all_targets):
			protein_dict[target] = process_target(target, "bond_cutoff")
		with open(r"protein_dict.pkl", "wb") as output_file:
			pickle.dump(protein_dict, output_file)
	else:
		protein_dict = pd.read_pickle("protein_dict.pkl")


if not os.path.isfile("combined_list.pkl"):
	if not os.path.isfile("ligand_dict.pkl"):
		print("Creating ligand dictionary")
		ligand_dict = {}
		for target in progressbar(all_targets):
			ligand_dict[target] = process_ligands(target)
			# Keep all the ligands which are active.
			ligand_dict[target][0]
			ligands = [
				ligand
				for ligand in ligand_dict[target]
				if ligand["targets"][0][0]
			]
			# Create a list of the decoys.
			neg_ligands = [
				ligand
				for ligand in ligand_dict[target]
				if not ligand["targets"][0][0]
			]
			# Add (a sample of) the decoys to the selected ligands list.
			ligands += np.random.choice(
				neg_ligands,
				size=len(ligands),
				replace=False,
			).tolist()
			ligand_dict[target] = ligands
		with open(r"ligand_dict.pkl", "wb") as output_file:
			pickle.dump(ligand_dict, output_file)
	else:
		ligand_dict = pd.read_pickle("ligand_dict.pkl")


def combine(protein, ligand):
	combined = {}
	combined["targets"] = ligand["targets"]
	combined["id"] = ligand["id"]
	combined["graph"] = protein["graph"] + ligand["graph"]
	combined["graph"] = [[entry[0] - 1, entry[1], entry[2] - 1] for entry in combined["graph"]]
	combined["node_features"] = protein["node_features"] + ligand["node_features"]
	return combined

if not os.path.isfile("combined_list.pkl"):
	print("Combining proteins and ligands")
	combined_list = []
	for target in progressbar(all_targets):
		for ligand in ligand_dict[target]:
			combined_list.append(combine(protein_dict[target], ligand))
	with open(r"combined_list.pkl", "wb") as output_file:
		pickle.dump(combined_list, output_file)
else:
	combined_list = pd.read_pickle("combined_list.pkl")

np.random.shuffle(combined_list)


def write_jsonl(name, start_index, end_index):
	with open(f"{name}.jsonl", "w") as outfile:
		for entry in progressbar(combined_list[start_index:end_index]):
			json.dump(entry, outfile)
			outfile.write('\n')

print(len(combined_list))

if not os.path.isfile("train.jsonl"):
	write_jsonl("train", 0, 2000)
if not os.path.isfile("valid.jsonl"):
	write_jsonl("valid", 2000, 2500)
if not os.path.isfile("test.jsonl"):
	write_jsonl("test", 2500, -1)
