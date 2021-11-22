# -*- coding: utf-8 -*-

import numpy as np
import torch


def data_lode(args):
	if args.t == 'o':
		drug_protein = np.load('data/drug_protein_indepent.npy')
	else:
		path = 'data/mat_drug_protein_' + args.t + '.txt'
		print(path)
		drug_protein = np.loadtxt(path)
		
	drug_disease = np.load('data/drug_disease.npy')
	drug_se = np.load('data/drug_se.npy')
	drug_drug = np.load('data/drug_drug.npy')
	
	protein_disease = np.load('data/protein_disease.npy')
	protein_protein = np.load('data/protein_protein.npy')
	
	protein_structure = torch.load('data/protein_structure.pt').requires_grad_(False)
	drug_structure = torch.load('data/drug_structure.pt').requires_grad_(False)
	
	drug_num = drug_disease.shape[0]
	protein_num = drug_protein.shape[1]
	disease_num = drug_disease.shape[1]
	se_num = drug_se.shape[1]

	DT_num = drug_num + protein_num
	
	count = [drug_num,protein_num,disease_num,se_num]
	num = drug_num + protein_num + disease_num + se_num
	
	pre_train_data = np.copy(drug_protein)
	np.random.seed(10)
	one_index = np.where(drug_protein == 1)
	one_index = np.stack(one_index).T
	np.random.shuffle(one_index)
	one_index = one_index.T
	one_num = int((one_index.shape[1])/args.kfold)
	if args.index==args.kfold:
		one_index = one_index[:,((args.index-1)*one_num):]
	else:
		one_index = one_index[:,((args.index-1)*one_num):(args.index*one_num)]
	pre_train_data[one_index[0], one_index[1]] = 0
	
	zero_index = np.where(drug_protein == 0)
	zero_index = np.stack(zero_index).T
	np.random.shuffle(zero_index)
	zero_index = zero_index.T
	zero_num = one_index.shape[1]
	zero_index = zero_index[:,:zero_num]
	
	test_label = np.concatenate((np.ones(one_index.shape[1]),np.zeros(one_index.shape[1])))

	A = np.zeros((10,num,num),dtype=np.int8)
	
	A[0,:drug_num,(drug_num):(drug_num + protein_num)] = pre_train_data
	A[1] = A[0].T
	A[2,:drug_num,(drug_num + protein_num):(drug_num + protein_num + disease_num)] = drug_disease
	A[3] = A[2].T
	A[4,:drug_num,(drug_num + protein_num + disease_num):] = drug_se
	A[5] = A[4].T
	A[6,drug_num:(drug_num + protein_num),(drug_num + protein_num):(drug_num + protein_num + disease_num)] = protein_disease
	A[7] = A[6].T
	A[8,:drug_num,:drug_num] = drug_drug
	A[9,(drug_num):(drug_num + protein_num),(drug_num):(drug_num + protein_num)] = protein_protein
	
	a = np.eye(num,dtype=np.int8)
	a = np.expand_dims(a,axis=0)
	A = np.concatenate((A,a))
	A = torch.from_numpy(A)
	
	train_label = torch.from_numpy(pre_train_data).float()
	
	DTI = np.zeros((DT_num,DT_num))
	DTI[:drug_num,drug_num:] = pre_train_data
	DTI[drug_num:,:drug_num] = pre_train_data.T
	DTI = torch.from_numpy(DTI).float()

	return(A, DTI, protein_structure, drug_structure, train_label, test_label, one_index, zero_index, count)

def load_uniquedata():
	unique = np.loadtxt('data/mat_drug_protein_unique.txt')

	drug_protein = np.load('data/drug_protein.npy')
	drug_disease = np.load('data/drug_disease.npy')
	drug_se = np.load('data/drug_se.npy')
	drug_drug = np.load('data/drug_drug.npy')
	
	protein_disease = np.load('data/protein_disease.npy')
	protein_protein = np.load('data/protein_protein.npy')
	
	protein_structure = torch.load('data/protein_structure.pt').requires_grad_(False)
	drug_structure = torch.load('data/drug_structure.pt').requires_grad_(False)
	
	drug_num = drug_disease.shape[0]
	protein_num = drug_protein.shape[1]
	disease_num = drug_disease.shape[1]
	se_num = drug_se.shape[1]

	DT_num = drug_num + protein_num
	
	count = [drug_num,protein_num,disease_num,se_num]
	num = drug_num + protein_num + disease_num + se_num
	
	one_index = np.where(unique==3)	
	zero_index = np.where(unique==2)
	
	one_index = np.stack(one_index)
	zero_index = np.stack(zero_index).T
	np.random.shuffle(zero_index)
	zero_index = zero_index.T
	zero_index = zero_index[:,:one_index[0].size]
	
	pre_train_data = np.copy(drug_protein)
	pre_train_data[one_index[0], one_index[1]] = 0
	
	test_label = np.concatenate((np.ones(one_index.shape[1]),np.zeros(one_index.shape[1])))

	A = np.zeros((10,num,num),dtype=np.int8)
	
	A[0,:drug_num,(drug_num):(drug_num + protein_num)] = pre_train_data
	A[1] = A[0].T
	A[2,:drug_num,(drug_num + protein_num):(drug_num + protein_num + disease_num)] = drug_disease
	A[3] = A[2].T
	A[4,:drug_num,(drug_num + protein_num + disease_num):] = drug_se
	A[5] = A[4].T
	A[6,drug_num:(drug_num + protein_num),(drug_num + protein_num):(drug_num + protein_num + disease_num)] = protein_disease
	A[7] = A[6].T
	A[8,:drug_num,:drug_num] = drug_drug
	A[9,(drug_num):(drug_num + protein_num),(drug_num):(drug_num + protein_num)] = protein_protein
	
	a = np.eye(num,dtype=np.int8)
	a = np.expand_dims(a,axis=0)
	A = np.concatenate((A,a))
	A = torch.from_numpy(A)
	
	train_label = torch.from_numpy(pre_train_data).float()
	
	DTI = np.zeros((DT_num,DT_num))
	DTI[:drug_num,drug_num:] = pre_train_data
	DTI[drug_num:,:drug_num] = pre_train_data.T
	DTI = torch.from_numpy(DTI).float()

	return(A, DTI, protein_structure, drug_structure, train_label, test_label, one_index, zero_index, count)
