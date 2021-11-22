import numpy as np

def get_metrics(real_score, predict_score):
	'''
	Parameters
	----------
	real_score : np.array
		shape(n,)
	predict_score : np.array
		shape(n,)

	Returns
	-------
	list
		[aupr, auc, f1_score, accuracy, recall, specificity, precision]
	'''
	real_score = real_score.flatten()
	predict_score = predict_score.flatten()
	
	real_score = np.mat(real_score)
	sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))  
	sorted_predict_score_num = len(sorted_predict_score)
	thresholds = sorted_predict_score[(np.array([sorted_predict_score_num])*np.arange(1, 1000)/np.array([1000])).astype(int)]
	thresholds = np.mat(thresholds)
	thresholds_num = thresholds.shape[1]

	predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
	negative_index = np.where(predict_score_matrix < thresholds.T)
	positive_index = np.where(predict_score_matrix >= thresholds.T)
	predict_score_matrix[negative_index] = 0
	predict_score_matrix[positive_index] = 1

	TP = predict_score_matrix*real_score.T
	FP = np.mat(predict_score_matrix.sum(axis=1)).T - TP
	FN = real_score.sum()-TP
	TN = len(real_score.T)-TP-FP-FN

	fpr = FP/(FP+TN)
	tpr = TP/(TP+FN)
	ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
	ROC_dot_matrix.T[0] = [0, 0]
	ROC_dot_matrix=np.c_[ROC_dot_matrix, [1, 1]]
	x_ROC = ROC_dot_matrix[0].T
	y_ROC = ROC_dot_matrix[1].T
	auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

	recall_list = tpr
	precision_list = TP/(TP+FP)
	PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
	PR_dot_matrix[1,:] = -PR_dot_matrix[1,:]
	PR_dot_matrix.T[0] = [0, 1]
	PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
	x_PR = PR_dot_matrix[0].T
	y_PR = PR_dot_matrix[1].T
	aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

	f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
	accuracy_list = (TP+TN)/len(real_score.T)
	specificity_list = TN/(TN+FP)

	max_index = np.argmax(f1_score_list)
	f1_score = f1_score_list[max_index, 0]
	accuracy = accuracy_list[max_index, 0]
	specificity = specificity_list[max_index, 0]
	recall = recall_list[max_index, 0]
	precision = precision_list[max_index, 0]
	return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]