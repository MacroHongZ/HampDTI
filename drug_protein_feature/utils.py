
from sklearn.metrics import roc_auc_score, average_precision_score


def get_metrics(real_score, predict_score):
	
	y_label = real_score.flatten()
	y_pred = predict_score.flatten()    

	return [round(roc_auc_score(y_label, y_pred),4), 
		    round(average_precision_score(y_label, y_pred),4)]