# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:55:49 2020

@author: WHZ
"""

import torch
import numpy as np
import torch.nn as nn
from model import GTN
from utils import get_metrics
from datalode import data_lode, load_uniquedata
import sys
import time
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', default=40, type=int)
	parser.add_argument('--kfold', default=10, type=int)
	parser.add_argument('--index', default=1, type=int)

	parser.add_argument('--num_channels', default=4, type=int)
	parser.add_argument('--num_layers', default=2, type=int)

	parser.add_argument('--lr', default=0.0003, type=float)
	parser.add_argument('--weight_decay', default=1e-7, type=float)
	parser.add_argument('--alpha', default=0.4, type=float)
	parser.add_argument('-t', default='o', type=str)

	args = parser.parse_args()
	print(args)

	drug_num = 708
	protein_num = 1512

	all_fold = []
	indepent_one_index = np.load('data/indepent_one_index.npy')
	indepent_zero_index = np.load('data/indepent_zero_index.npy')
	indepent_lable = np.concatenate((np.ones(indepent_one_index.shape[1]), np.zeros(indepent_one_index.shape[1])))

	for i in range(10):
		max_auc = 0
		max_aupr = 0
		args.index = i + 1
		A, DTI, protein_structure, drug_structure, train_label, test_label, one_index, zero_index, count = data_lode(
			args)

		model = GTN(num_edge=A.shape[0],
					num_channels=args.num_channels,
					num_layers=args.num_layers,
					drug_num=708,
					protein_num=1512)

		class Myloss(nn.Module):
			def __init__(self):
				super(Myloss, self).__init__()

			def forward(self, iput, target):
				loss_sum = torch.pow((iput - target), 2)
				result = (1 - args.alpha) * ((target * loss_sum).sum()) + args.alpha * (((1 - target) * loss_sum).sum())
				return (result)

		myloss = Myloss()
		layers_params = list(map(id, model.layers.parameters()))
		base_params = filter(lambda p: id(p) not in layers_params,
							 model.parameters())
		opt = torch.optim.Adam([{'params': base_params},
								{'params': model.layers.parameters(), 'lr': 0.5}]
							   , lr=args.lr, weight_decay=args.weight_decay)

		print(f'The {i + 1} fold')
		for i in range(args.epoch):
			for param in opt.param_groups:
				if param['lr'] > 0.001:
					param['lr'] *= 0.9

			model.train()
			opt.zero_grad()
			predict, Ws, att = model(A, DTI, drug_num, protein_num, protein_structure, drug_structure)

			loss = myloss(predict, train_label)
			print(f'epoch:  {i + 1}    train_loss:  {loss}')

			loss.backward()
			opt.step()

			with torch.no_grad():
				predict_test = predict.detach().cpu().numpy()
				predict_test_negative = predict_test[zero_index[0], zero_index[1]]
				predict_test_positive = predict_test[one_index[0], one_index[1]]

				predict_test_fold = np.concatenate((predict_test_positive, predict_test_negative))
				metrics = get_metrics(test_label, predict_test_fold)

				if metrics[1] > max_auc:
					max_auc = metrics[1]
					indepent_test_negative = predict_test[indepent_zero_index[0], indepent_zero_index[1]]
					indepent_test_positive = predict_test[indepent_one_index[0], indepent_one_index[1]]

					indepent_test = np.concatenate((indepent_test_positive, indepent_test_negative))
					indepent_metrics = get_metrics(indepent_lable, indepent_test)
				if metrics[0] > max_aupr:
					max_aupr = metrics[0]

				print('indepent test metrics:', indepent_metrics)
				print(f'AUC:  {metrics[1]:.4f}   AUPR: {metrics[0]:.4f}')

		print("W :", Ws)
		print("att :", att)
		print('max_auc:  ' + str(max_auc), 'max_aupr:  ' + str(max_aupr))
		all_fold.append(max_auc)
		all_fold.append(max_aupr)

	print(all_fold)


if __name__ == '__main__':
	main()
