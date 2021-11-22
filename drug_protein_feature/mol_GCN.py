# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:41:12 2020

@author: WHZ
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool as gep, global_max_pool as gmp, global_add_pool


# GCN based model
class PretrainModel(torch.nn.Module):
	def __init__(self, n_output=1, num_features_mol=78, num_features_pro=261, output_dim=128, dropout=0.2):
		super(PretrainModel, self).__init__()

		print('PretrainModel Loaded')
		self.n_output = n_output
		self.mol_conv1 = GCNConv(num_features_mol, output_dim)
		
		self.pro = nn.Sequential(nn.Linear(num_features_pro, num_features_pro),
					            nn.ReLU(),
					            nn.Linear(num_features_pro, 256),
					            nn.ReLU(),
					            nn.Linear(256, output_dim))

		self.relu = nn.ReLU()

	def forward(self, data_mol, squence):
		
		mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
	   
		x = self.mol_conv1(mol_x, mol_edge_index)
		x = self.relu(x)

		x = gmp(x, mol_batch)  # global pooling

		x2 = self.pro(squence)
		y = torch.mm(x, x2.t())
# 		torch.save(x, 'drug_structure.pt')
# 		torch.save(x2, 'protein_structure.pt')
		
		y = torch.sigmoid(y)
		
		return y
