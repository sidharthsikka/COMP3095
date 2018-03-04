import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool
import pickle
from operator import itemgetter

def rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())

def plotting_shock_amplification(row, amplification_post, amplification_pre, shock):
	s = "Amplifcation vs "
	s+=shock
	plt.subplot(2, 1, 1)
	plt.scatter(row, amplification_post)
	plt.xticks(row, row, rotation='vertical')
	plt.title(s)
	plt.xlabel('Industries')
	plt.ylabel('Amplification post shock')

	plt.subplot(2, 1, 2)
	plt.scatter(row, amplification_pre)
	plt.xlabel('Industries')
	plt.ylabel('Amplification pre shock')

	plt.show()

def model_intialization(df_2000):
	# in strength of a node which is calculated for all the money coming into it
	in_strength = []
	for row in df_2000.iterrows():
		index, data = row
		in_strength.append(sum(data.tolist()))

	# out strength of a node which is calculated for all the money going out of a node
	out_strength = []
	for col in df_2000.columns:
		out_strength.append(df_2000[col].sum())

	# combines i and o to give a propensity to spend for a sector by putting o/i if i>=o else 1
	alpha = []
	for i in range(len(df_2000.columns)):
		if in_strength[i] >= out_strength[i]:
			if in_strength[i] != 0:  # ASSUMPTION
				alpha.append(out_strength[i] / in_strength[i])
			else:
				alpha.append(1)
		else:
			alpha.append(1)

	# again combines i and o to give a borrowing capacity of a company by putting o-i if o>i else 0
	beta = []
	for i in range(len(df_2000.columns)):
		if out_strength[i] > in_strength[i]:
			beta.append(out_strength[i] - in_strength[i])
		else:
			beta.append(0)

	# for each connection i-j we normalize it by dividing the money going out to j by the total money going out
	m = []
	y = df_2000.columns
	for i in range(len(out_strength)):
		if out_strength[i] != 0:  # ASSUMPTION
			m.append((df_2000[y[i]] / out_strength[i]).tolist())
		else:
			m.append(df_2000[y[i]])

	# INITIAL M & E
	M = np.matmul(np.diag(out_strength), m)
	one = [1 for _ in range(len(df_2000.columns))]
	E = np.matmul(np.matmul(np.diag(alpha), np.transpose(M)), one) + beta
	return alpha, beta, one, m, M, E

def node_deletion_shock(temp):
	alpha, beta, one, m, M, E, start, end = temp
	amplification = []
	#TODO ADD IN VULNERABILITY CALCULATION FOR EACH NODE
	for i in range(start, end+1):
		M_1 = copy.deepcopy(M)
		M_1[i] = [0 for _ in range(len(M_1[i]))]
		new_m = copy.deepcopy(m)
		new_m[i] = [0 for _ in range(len(new_m[i]))]
		alpha[i] = 0
		for j in range(len(M_1)):
			M_1[j][i] = 0
			new_m[j][i] = 0
		M_next = M_1
		E_prev = E
		E_initial_shock = np.matmul(np.matmul(np.diag(alpha), np.transpose(M_next)),one) + beta
		E_initial_loss = E_initial_shock - E
		if(sum(E_initial_loss) != 0):
			for j in range(1, 100):
				E_next = np.matmul(np.matmul(np.diag(alpha), np.transpose(M_next)),one) + beta
				M_next = np.matmul(np.diag(E_next), new_m)
				rms = rmse(np.array(E_next), np.array(E_prev))
				if rms <= 13150:
					break
				E_prev = E_next
			expenditure_difference = E_next - E
			amplification.append(sum(expenditure_difference)/sum(E_initial_loss))
		else:
			amplification.append(1)
	return amplification

def start_node_deletion(df):
	alpha, beta, one, m, M, E = model_intialization(df)
	start = [0,250,500,750,1000,1250,1500,1750,2000,2250]
	end = [249,499,749,999,1249,1499,1749,1999,2249,2463]
	iterable = list()
	for i in range(len(start)):
		iterable.append((alpha, beta, one, m, M, E, start[i], end[i]))
	pool = Pool(processes=5)
	results = pool.map(node_deletion_shock, iterable)
	with open('results_amp_vul.pickle', 'wb') as f:
		pickle.dump(results, f)

def get_importance(y,E):
	importance_table = list()
	for i in range(len(E)):
		importance_table.append((y[i],E[i]))
	importance_table.sort(key=lambda tup: tup[1])
	return importance_table


if __name__ == "__main__":
	df_2000 = pd.read_pickle('WIOD_Data/pickled_data/2000.pkl')
	alpha, beta, one, m, M, E = model_intialization(df_2000)
	print(E[110:114])
	results = node_deletion_shock((alpha, beta, one, m, M, E,110,114))
	print(results)
	# results = pickle.load(open("results.pickle", "rb"))
	# results_post = list()
	# results_pre = list()
	# for tup in results:
	# 	results_post.append(tup[0])
	# 	results_pre.append(tup[1])
	# flatten_post = [item for sublist in results_post for item in sublist]
	# flatten_pre = [item for sublist in results_pre for item in sublist]
	#TODO COMBINE RESULTS INTO ONE AND MERGE WITH VULNERABILITY PRODUCING A BUBBLE PLOT WITH A SIZING ACCORDING TO IMPORTANCE

	# Trial with first five nodes
