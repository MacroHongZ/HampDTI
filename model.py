# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:55 2020

@author: WHZ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GTN(nn.Module):

	def __init__(self,num_edge,num_channels,num_layers,drug_num,protein_num):
		super(GTN, self).__init__()
		self.num_edge = num_edge
		self.num_channels = num_channels
		self.num_layers = num_layers

		layers = []
		for i in range(num_layers):
			if i == 0:
				layers.append(GTLayer(num_edge, num_channels, first=True))
			else:
				layers.append(GTLayer(num_edge, num_channels, first=False))
		self.layers = nn.ModuleList(layers)
		
		self.linear_d1 = nn.Linear(128,128)
		self.linear_d2 = nn.Linear(128,128)
		self.linear_d3 = nn.Linear(128,128)
		
		self.linear_p1 = nn.Linear(128,128)
		self.linear_p2 = nn.Linear(128,128)
		self.linear_p3 = nn.Linear(128,128)
						
		self.a = nn.Parameter(torch.Tensor((num_channels+1),1,1))
		nn.init.constant_(self.a, 1)
		
	def SGC(self, feature, adj):
			
		adj = adj + (torch.eye(adj.shape[0]))*2
		deg = torch.sum(adj, dim=1)
		deg[deg<=1e-10]=1
		deg_inv = deg.pow(-0.5)
		deg_inv = deg_inv*torch.eye(adj.shape[0]).type(torch.FloatTensor)
		adj = torch.mm(deg_inv,adj)
		adj = torch.mm(adj, deg_inv).type(torch.FloatTensor)
	
		output = torch.mm(adj, feature)
	
		return output
	
	def normalization(self, H):
		for i in range(self.num_channels):
			if i==0:
				H_ = self.norm(H[i]).unsqueeze(0)
			else:
				H_ = torch.cat((H_,self.norm(H[i]).unsqueeze(0)), dim=0)
		return H_

	def norm(self, H, add=False):

		H = H + (torch.eye(H.shape[0]))
		deg = torch.sum(H, dim=1)
		deg[deg<=1e-10]=1
		deg_inv = deg.pow(-1)
		deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
		H = torch.mm(deg_inv,H)
		return H
	
	def forward(self, A, DTI, drug_num, protein_num, protein_structure, drug_structure):
		A = A.unsqueeze(0)
		
		drug = drug_structure
		protein = protein_structure
		
		drug1 = F.relu(self.linear_d1(drug))
		drug2 = F.relu(self.linear_d2(drug1))
		drug3 = F.relu(self.linear_d3(drug2))
		
		protein1 = F.relu(self.linear_p1(protein))
		protein2 = F.relu(self.linear_p2(protein1))
		protein3 = F.relu(self.linear_p3(protein2))
		
		feature = torch.cat((drug3,protein3),dim=0)
		
# auto-metapath
		Ws = []
		for i in range(self.num_layers):
			if i == 0:
				H, W = self.layers[i](A)
			else:
				H = self.normalization(H)
				H, W = self.layers[i](A, H)
			Ws.append(W)

# SGCN			
		adj1 = DTI
		adj = H[:,:(drug_num+protein_num),:(drug_num+protein_num)]
		
		X_conv1 = self.SGC(feature, adj1)
		X_conv1 = self.SGC(X_conv1, adj1)
		
		for i in range(self.num_channels):
			if i == 0:
				X_conv2 = self.SGC(feature, adj[i])
				X_conv2 = self.SGC(X_conv2, adj[i])
			else:
				X_tem = self.SGC(feature, adj[i])
				X_tem = self.SGC(X_tem, adj[i])
				if i == 1:
					X_conv3 = torch.stack((X_conv2, X_tem))
				else:
					X_conv3 = torch.cat((X_tem.unsqueeze(0),X_conv3),dim=0)
				
		X_conv = torch.cat((X_conv1.unsqueeze(0),X_conv3),dim=0)

# Multi-path weight sum		
		att = F.softmax(self.a, dim=0)
		conv_sum = torch.sum((att * X_conv), dim=0)
		
		drug_feature = conv_sum[:drug_num]
		protein_feature = conv_sum[drug_num:]

# MF prediction		
		y = torch.mm(drug_feature, protein_feature.t())
		y = torch.sigmoid(y)

		return(y, Ws, att)

class GTLayer(nn.Module):
	
	def __init__(self, in_channels, out_channels, first=True):
		super(GTLayer, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.first = first
		if self.first == True:
			self.conv1 = GTConv(in_channels, out_channels)
			self.conv2 = GTConv(in_channels, out_channels)
		else:
			self.conv1 = GTConv(in_channels, out_channels)
	
	def forward(self, A, H_=None):
		if self.first == True:
			a = self.conv1(A)
			b = self.conv2(A)
			H = torch.bmm(a,b)
			W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
		else:
			a = self.conv1(A)
			H = torch.bmm(H_,a)
			W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
		return H,W

class GTConv(nn.Module):
	
	def __init__(self, in_channels, out_channels):
		super(GTConv, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
		self.bias = None
		self.reset_parameters()
	def reset_parameters(self):
		nn.init.normal_(self.weight)
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, A):
		A = torch.sum(A*(F.softmax(self.weight, dim=1)), dim=1)
		return A

