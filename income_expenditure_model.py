import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool
import pickle

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def plotting_shock_amplification(df_2000, amplification_post, amplification_pre, shock): # impact(expenditure) vs vulnerability(how much changed when others shocked) with circle indicated its importance(trade value)
    plt.subplot(2, 1, 1)
    plt.plot(df_2000.columns, amplification_post, 'o-')
    plt.title('Amplification VS ', shock)
    plt.xlabel('Industries')
    plt.ylabel('Amplification')

    plt.subplot(2, 1, 2)
    plt.plot(df_2000.columns, amplification_pre, '.-')
    plt.xlabel('Industries')
    plt.ylabel('Amplification')

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
                alpha.append(out_strength[i])
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

def edge_deformation_shock(temp):
	alpha, beta, one, m, M, E, start, end, degrade = temp # start and end here are pairs of nodes
	print("START EDGE DEGREDATION SHOCKS FOR NODE PAIR ", start, " AND ", end)
	M_T = list()
	M_T.append(M)
	E_T = list()
	E_T.append(E)
	M_1 = copy.deepcopy(M)
	M_1[i][j] = M_1[i][j] * degrade
	M_T.append(M_1)
	for j in range(1, 100):
	   	E_T.append(np.matmul(np.matmul(np.diag(alpha), np.transpose(M_T[j])),one) + beta)
	   	M_T.append(np.matmul(np.diag(E_T[j]), m))
	   	rms = rmse(np.array(E_T[j]), np.array(E_T[j-1]))
	   	if rms <= 10:
	   		break
	expenditure_difference_post = E_T[len(E_T)-1] - E_T[1] #in comparison after initial shock has been applied
	expenditure_difference_pre = E_T[len(E_T)-1] - E_T[0] #in comparison before initial shock has been applied
	print("FINISHED EDGE DEGREDATION SHOCKS FOR NODE PAIR ", start, " AND ", end)
	return sum(expenditure_difference_post), sum(expenditure_difference_pre)

def bilateral_trade_deletion_shock(temp):
	# TODO Similar to edge deformation but instead of degrading a particular edge we completely remove any edges between a pair and adjust m accordingly keeping alpha and beta the same
	alpha, beta, one, m, M, E, start, end, degrade = temp # start and end here are pairs of nodes
	print("START EDGE DEGREDATION SHOCKS FOR NODE PAIR ", start, " AND ", end)
	M_T = list()
	M_T.append(M)
	E_T = list()
	E_T.append(E)
	M_1 = copy.deepcopy(M)
	M_1[i][j] = 0
	M_1[j][i] = 0
	m[i][j] = 0
	m[j][i] = 0
	M_T.append(M_1)
	for j in range(1, 100):
	   	E_T.append(np.matmul(np.matmul(np.diag(alpha), np.transpose(M_T[j])),one) + beta)
	   	M_T.append(np.matmul(np.diag(E_T[j]), m))
	   	rms = rmse(np.array(E_T[j]), np.array(E_T[j-1]))
	   	if rms <= 10:
	   		break
	expenditure_difference_post = E_T[len(E_T)-1] - E_T[1] #in comparison after initial shock has been applied
	expenditure_difference_pre = E_T[len(E_T)-1] - E_T[0] #in comparison before initial shock has been applied
	print("FINISHED EDGE DEGREDATION SHOCKS FOR NODE PAIR ", start, " AND ", end)
	return sum(expenditure_difference_post), sum(expenditure_difference_pre)

def node_deformation_shock(temp):
	alpha, beta, one, m, M, E, start, end, degrade_row, degrade_col = temp
	print("START NODE DELETION SHOCKS FOR NODE ", start, " AND ", end)
	amplification_post = [] # for each shock this stores the amplification score for plotting
	amplification_pre = []
	for i in range(start, end+1):
		M_T = list()
		M_T.append(M)
		E_T = list()
		E_T.append(E)
		M_1 = copy.deepcopy(M)
		M_1[i] = M_1[i] * degrade_row
		new_m = copy.deepcopy(m)
		new_m[i] = new_m[i] * degrade_row
		alpha[i] = alpha[i] * degrade_col/degrade_row
		for j in range(len(M_1)):
	   		M_1[j][i] = M_1[j][i] * degrade_col
	   		new_m[j][i] = new_m[j][i] * degrade_col
		M_T.append(M_1)
		for j in range(1, 100):
	   		E_T.append(np.matmul(np.matmul(np.diag(alpha), np.transpose(M_T[j])),one) + beta)
	   		M_T.append(np.matmul(np.diag(E_T[j]), new_m))
	   		rms = rmse(np.array(E_T[j]), np.array(E_T[j-1]))
	   		if rms <= 10:
	   			print("REACHED IN ", j)
	   			break
		expenditure_difference_post = E_T[len(E_T)-1] - E_T[1] #in comparison after initial shock has been applied
		expenditure_difference_pre = E_T[len(E_T)-1] - E_T[0] #in comparison before initial shock has been applied
		amplification_post.append(sum(expenditure_difference_post))
		amplification_pre.append(sum(expenditure_difference_pre))
	print("FINISHED NODE DELETION SHOCKS FOR NODE ", start, " AND ", end)
	return amplification_post, amplification_pre

def node_deletion_shock(temp):
	alpha, beta, one, m, M, E, start, end = temp
	print("START NODE DELETION SHOCKS FOR NODE ", start, " AND ", end)
	amplification_post = [] # for each shock this stores the amplification score for plotting
	amplification_pre = []
	for i in range(start, end+1):
		M_T = list()
		M_T.append(M)
		E_T = list()
		E_T.append(E)
		M_1 = copy.deepcopy(M)
		M_1[i] = [0 for _ in range(len(M_1[i]))]
		new_m = copy.deepcopy(m)
		new_m[i] = [0 for _ in range(len(new_m[i]))]
		alpha[i] = 0
		for j in range(len(M_1)):
	   		M_1[j][i] = 0
	   		new_m[j][i] = 0
		M_T.append(M_1)
		for j in range(1, 100):
	   		E_T.append(np.matmul(np.matmul(np.diag(alpha), np.transpose(M_T[j])),one) + beta)
	   		M_T.append(np.matmul(np.diag(E_T[j]), new_m))
	   		rms = rmse(np.array(E_T[j]), np.array(E_T[j-1]))
	   		if rms <= 0.01:
	   			print("REACHED IN ", j)
	   			break
		expenditure_difference_post = E_T[len(E_T)-1] - E_T[1] #in comparison after initial shock has been applied
		expenditure_difference_pre = E_T[len(E_T)-1] - E_T[0] #in comparison before initial shock has been applied
		amplification_post.append(sum(expenditure_difference_post))
		amplification_pre.append(sum(expenditure_difference_pre))
	print("FINISHED NODE DELETION SHOCKS FOR NODE ", start, " AND ", end)
	return amplification_post, amplification_pre

if __name__ == "__main__":
	df_2000 = pd.read_pickle('WIOD_Data/pickled_data/2000.pkl')

	# Delete any self pointing edges
	for row in df_2000.iterrows():
		index, data = row
		for col in df_2000.columns:
			if index == col:
				df_2000.loc[index, col] = 0

	# Delete any country where there is no trade
	for col in df_2000.columns:
   		if df_2000[col].sum() == 0 and sum(df_2000.loc[col].tolist()) == 0:
   			df_2000.drop(col, axis=1)
   			df_2000.drop(col, axis=0)

	# Original Data is expressed in millions of dollars so this will bring them all to dollars
	for col in df_2000.columns:
		df_2000[col] = df_2000[col] * 1000000

	print("TRYING MULTIPROCESSING")
	alpha, beta, one, m, M, E = model_intialization(df_2000)
	start = [0,500,1000,1500]
	end = [499,999,1499,2463]
	iterable = list()
	for i in range(4):
		iterable.append((alpha, beta, one, m, M, E, start[i], end[i]))
	pool = Pool(processes=4)
	results = pool.map(node_deletion_shock, iterable)
	with open('results.pickle', 'wb') as f:
		pickle.dump(results, f)
    '''G = nx.DiGraph()
    for i in df_2000.columns.values:
    G.add_node(i)

    weight_list = []

    for i in df_2000.columns.values:
        for j in df_2000.columns.values:
            if i != j and df_2000.loc[i,j]>0:
                weight_list.append((i,j,df_2000.loc[i,j]))

    G.add_weighted_edges_from(weight_list)
    nx.draw_spring(G)
    print("DRAWN")
    plt.savefig('2000_SPRING.png')'''