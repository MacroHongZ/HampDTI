# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:14:47 2020

@author: WHZ
"""

import numpy as np
from rdkit import Chem
import networkx as nx
import torch
from matplotlib import pyplot as plt

def atom_features(atom):
	# 44 +11 +11 +11 +1
	return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
										  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
										   'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
										   'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
										   'Pt', 'Hg', 'Pb', 'X']) +
					one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
					one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
					one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
					[atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
	if x not in allowable_set:
		# print(x)
		raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
	return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
	'''Maps inputs not in the allowable set to the last element.'''
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
	mol = Chem.MolFromSmiles(smile)

	c_size = mol.GetNumAtoms()

	features = []
	for atom in mol.GetAtoms():
		feature = atom_features(atom)
		features.append(feature / sum(feature))

	edges = []
	for bond in mol.GetBonds():
		edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
	g = nx.Graph(edges).to_directed()
 # 	nx.draw(g)
 # 	plt.show()

	edge_index = []
	mol_adj = np.zeros((c_size, c_size))
	for e1, e2 in g.edges:
		mol_adj[e1, e2] = 1
		# edge_index.append([e1, e2])
	mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
	index_row, index_col = np.where(mol_adj >= 0.5)
	for i, j in zip(index_row, index_col):
		edge_index.append([i, j])
	# print('smile_to_graph')
	# print(np.array(features).shape)
	features = torch.tensor(features).float()	#shape(c_size, 78)
	edge_index = torch.tensor(edge_index).t()	#shape(2, edge_num)
	
	return c_size, features, edge_index

