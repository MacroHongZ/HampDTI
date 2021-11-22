# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:22:38 2020

@author: WHZ
"""

import torch
import numpy as np
import torch.nn as nn
from mol_GCN import PretrainModel
from smiles_trans_graph import smile_to_graph
from utils import get_metrics
from dataload import collate, GraphDataset


def main():
	lr = 0.001
	weight_decay = 0.0001
	epoch = 300
	kfold = 10
	
	drug_protein = np.load('drug_protein.npy')
	drug_num = drug_protein.shape[0]
	protein_num = drug_protein.shape[1]
	
	label = np.copy(drug_protein)
	np.random.seed(10)
	one_index = np.where(drug_protein == 1)
	one_index = np.stack(one_index).T
	np.random.shuffle(one_index)
	one_index = one_index.T
	one_num = int((one_index.shape[1])/kfold)
	one_index = one_index[:,:one_num]
	label[one_index[0], one_index[1]] = 0
	
	zero_index = np.where(drug_protein == 0)
	zero_index = np.stack(zero_index).T
	np.random.shuffle(zero_index)
	zero_index = zero_index.T
	zero_num = one_num
	zero_index = zero_index[:,:zero_num]
	
	test_label = test_label = np.concatenate((np.ones(one_num),np.zeros(one_num)))
		
	smiles = []
	with open('smlies.txt') as dr:
		for s in dr:
			smiles.append(s.strip())
	
	protein = np.load('protein.npy')
	protein = torch.from_numpy(protein).float()
	
	
	model = PretrainModel()
	myloss = nn.BCELoss()
	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	
	
	data_mol = list(map(smile_to_graph,smiles))		
	
	drug_graphs_Data = GraphDataset(root='data', dataset="davis", graphs_dict=data_mol, dttype="drug")
	drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=708)

	profea = protein
	y = torch.tensor(label, dtype=torch.float32)
	for i in range(epoch):
		for data_mol in drug_graphs_DataLoader:
			model.zero_grad()
			predict = model(data_mol, profea)
			
			metrics_train = get_metrics(y.numpy(), predict.detach().cpu().numpy())
			print(metrics_train)
			loss = myloss(predict, y)
			loss.backward()
			opt.step()
			print(f'epoch:  {i+1}    train_loss:  {loss}')
					
		model.eval()
		with torch.no_grad():
			predict_test = predict.detach().cpu().numpy()
			
			predict_test_negative = predict_test[zero_index[0], zero_index[1]]
			predict_test_positive = predict_test[one_index[0], one_index[1]]

			predict_test = np.concatenate((predict_test_positive,predict_test_negative))
			metrics = get_metrics(test_label, predict_test)
			
			print(f'AUC:  {metrics[0]:.4f}   AUPR: {metrics[1]:.4f}')

if __name__ == '__main__':
	main()
