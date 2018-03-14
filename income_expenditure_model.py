import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool
import pickle
from operator import itemgetter
from networkx.algorithms import approximation as approx
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import random
import collections
plotly.tools.set_credentials_file(username='sidharthsikka', api_key='qb4RJVlP2OJQm41AUpcE')

def rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())

def model_intialization(df_2000):
	print()
	#Intial M
	M = list()
	for col in df_2000.columns:
		M.append(df_2000[col].values)

	# in strength of a node which is calculated for all the money coming into it
	count = 0.0
	in_strength = list()
	for i in range(len(M)):
		for j in range(len(M[0])):
			count += M[j][i]
		in_strength.append(count)
		count = 0.0

	# out strength of a node which is calculated for all the money going out of a node
	count = 0.0
	out_strength = list()
	for i in range(len(M)):
		for j in range(len(M[0])):
			count += M[i][j]
		out_strength.append(count)
		count = 0.0

	# combines i and o to give a propensity to spend for a sector by putting o/i if i>=o else 1
	alpha = list()
	for i in range(len(df_2000.columns)):
		if in_strength[i] >= out_strength[i]:
			if in_strength[i] != 0.0:  # ASSUMPTION
				alpha.append(out_strength[i] / in_strength[i])
			else:
				alpha.append(1.0)
		else:
			alpha.append(1.0)

	# again combines i and o to give a borrowing capacity of a company by putting o-i if o>i else 0
	beta = list()
	for i in range(len(df_2000.columns)):
		if out_strength[i] > in_strength[i]:
			beta.append(out_strength[i] - in_strength[i])
		else:
			beta.append(0.0)

	# for each connection i-j we normalize it by dividing the money going out to j by the total money going out
	m = list()
	for i in range(len(out_strength)):
		if out_strength[i] != 0.0:  # ASSUMPTION
			m.append(M[i]/out_strength[i])
		else:
			m.append(M[i])
	# INITIAL E
	one = [1.0 for _ in range(len(df_2000.columns))]
	E = np.matmul(np.matmul(np.diag(alpha), np.transpose(M)), one) + beta
	return alpha, beta, one, m, M, E

def bilateral_trade_deletion_shock(temp):
	alpha, beta, one, m, M, E, source, end = temp
	amplification = list()
	vulnerability = list()
	vulnerability = [0 for _ in range(len(E))]
	M_next = copy.deepcopy(M)
	M_next[source][end] = 0
	M_next[end][source] = 0
	new_m = copy.deepcopy(m)
	new_m[source][end] = 0
	new_m[end][source] = 0
	E_prev = E
	alpha_diag = np.diag(alpha)
	E_initial_shock = np.matmul(np.matmul(alpha_diag, np.transpose(M_next)),one) + beta
	E_initial_loss = E_initial_shock - E
	if(sum(E_initial_loss) != 0):
		for j in range(1, 100):
			E_next = np.matmul(np.matmul(alpha_diag, np.transpose(M_next)),one) + beta
			M_next = np.matmul(np.diag(E_next), new_m)
			rms = rmse(np.array(E_next), np.array(E_prev))
			if rms <= 0.1:
				break
			E_prev = E_next
		expenditure_difference = E_next - E
		for k in range(len(E)):
			if(E[k] != 0) and (k==source or k==end):
				vulnerability[k] += 1-E_next[k]/E[k]
		amplification.append(sum(expenditure_difference)/sum(E_initial_loss))
	else:
		amplification.append(0)
	return amplification, vulnerability

def node_deformation_shock(temp):
	alpha, beta, one, m, M, E, start, end, degrade_row, degrade_col = temp
	amplification = list()
	vulnerability = list()
	vulnerability = [0 for _ in range(len(E))]
	for i in range(start, end+1):
		print("NODE ", i , " BEING DELETED")
		M_next = copy.deepcopy(M)
		M_next[i] = M_next[i] * degrade_row
		new_m = copy.deepcopy(m)
		new_m[i] = new_m[i] * degrade_row
		new_alpha = copy.deepcopy(alpha)
		new_alpha[i] = new_alpha[i] * (degrade_col/degrade_row)
		for j in range(len(M_next)):
			M_next[j][i] = M_next[j][i] * degrade_col
			new_m[j][i] = new_m[j][i] * degrade_col
		E_prev = E
		new_alpha_diag = np.diag(new_alpha)
		E_initial_shock = np.matmul(np.matmul(new_alpha_diag, np.transpose(M_next)),one) + beta
		E_initial_loss = E_initial_shock - E
		if(sum(E_initial_loss) != 0):
			for j in range(1, 100):
				E_next = np.matmul(np.matmul(new_alpha_diag, np.transpose(M_next)),one) + beta
				M_next = np.matmul(np.diag(E_next), new_m)
				rms = rmse(np.array(E_next), np.array(E_prev))
				if rms <= 0.1:
					print(j)
					break
				E_prev = E_next
			expenditure_difference = E_next - E
			for k in range(len(E)):
				if(E[k] != 0 and k!=i):
					vulnerability[k] += 1-E_next[k]/E[k]
			amplification.append(sum(expenditure_difference)/sum(E_initial_loss))
		else:
			amplification.append(0)
	return amplification, vulnerability

def node_deletion_shock(temp):
	alpha, beta, one, m, M, E, start, end = temp
	amplification = list()
	vulnerability = list()
	vulnerability = [0 for _ in range(len(E))]
	for i in range(start, end+1):
		print("NODE ", i , " BEING DELETED")
		M_next = copy.deepcopy(M)
		M_next[i] = [0 for _ in range(len(M_next[i]))]
		new_m = copy.deepcopy(m)
		new_m[i] = [0 for _ in range(len(new_m[i]))]
		new_alpha = copy.deepcopy(alpha)
		new_alpha[i] = 0
		for j in range(len(M_next)):
			M_next[j][i] = 0
			new_m[j][i] = 0
		E_prev = E
		new_alpha_diag = np.diag(new_alpha)
		E_initial_shock = np.matmul(np.matmul(new_alpha_diag, np.transpose(M_next)),one) + beta
		E_initial_loss = E_initial_shock - E
		if(sum(E_initial_loss) != 0):
			for j in range(1, 100):
				E_next = np.matmul(np.matmul(new_alpha_diag, np.transpose(M_next)),one) + beta
				M_next = np.matmul(np.diag(E_next), new_m)
				rms = rmse(np.array(E_next), np.array(E_prev))
				if rms <= 0.1:
					break
				E_prev = E_next
			expenditure_difference = E_next - E
			for k in range(len(E)):
				if(E[k] != 0 and k!=i):
					vulnerability[k] += 1-E_next[k]/E[k]
			amplification.append(sum(expenditure_difference)/sum(E_initial_loss))
		else:
			amplification.append(0)
	return amplification, vulnerability

def start_node_deletion(df):
	alpha, beta, one, m, M, E = model_intialization(df)
	start = [x for x in range(0,len(E)-200,200)]
	end = [x for x in range(199,len(E)-1,200)]
	end[len(end)-1] = len(E)-1
	iterable = list()
	for i in range(len(start)):
		iterable.append((alpha, beta, one, m, M, E, start[i], end[i]))
	pool = Pool(processes=5)
	results = pool.map(node_deletion_shock, iterable)
	with open('results_amp_vul.pickle', 'wb') as f:
		pickle.dump(results, f)

def start_node_deformation(df):
	alpha, beta, one, m, M, E = model_intialization(df)
	start = [x for x in range(0,len(E)-200,200)]
	end = [x for x in range(199,len(E)-1,200)]
	end[len(end)-1] = len(E)-1
	iterable = list()
	for i in range(len(start)):
		iterable.append((alpha, beta, one, m, M, E, start[i], end[i],0.4,0.3))
	pool = Pool(processes=5)
	results = pool.map(node_deformation_shock, iterable)
	with open('deformation_amp_vul.pickle', 'wb') as f:
		pickle.dump(results, f)

def get_importance(y,E):
	importance_table = list()
	for i in range(len(E)):
		importance_table.append((y[i],E[i]))
	importance_table.sort(key=lambda tup: tup[1])
	return importance_table

def aggregated_node_deletion():
	years = list(range(2000,2015))
	for year in years:
		df_2000 = pd.read_pickle('WIOD_Data/pickled_data/' + str(year) + '.pkl')
		countries = []
		countries_index = []
		i = 0
		for col in df_2000.columns:
			temp = col.split(',')
			if not(temp[len(temp)-1].lstrip() in  countries):
				countries.append(temp[len(temp)-1].lstrip())
				countries_index.append(i)
			i+=1
		countries_index.append(len(df_2000.columns))
		data_agg = []
		for row in df_2000.iterrows():
			index, data = row
			data_agg_temp = []
			for ind in range(len(countries_index)-1):
				data_agg_temp.append(sum(data[countries_index[ind]:countries_index[ind+1]]))
			data_agg.append(data_agg_temp)
		agg = []
		for ind in range(len(countries_index)-1):
			agg.append(np.sum(data_agg[countries_index[ind]:countries_index[ind+1]], axis=0))
		df = pd.DataFrame(agg,columns=countries,index=countries)
		for col in df.columns:
			df.loc[col,col] = 0.0
		alpha, beta, one, m, M, E = model_intialization(df)
		results = node_deletion_shock((alpha, beta, one, m, M, E,0,len(E)-1))
		size = np.array(E)/np.min(E[np.nonzero(E)])
		amplification = list()
		vulnerability = list()
		flatten_amp = results[0]
		flatten_vul = results[1]
		for i in range(len(flatten_vul)):
			flatten_vul[i] /= len(E)
		# plt.scatter(E, flatten_vul)
		# plt.title('Vulnerability vs Expenditure')
		# plt.xlabel('Expenditure')
		# plt.ylabel('Vulnerability')
		# plt.savefig(str(year) + '_agg_evsvul.png')
		# plt.clf()
		temp = zip(flatten_vul, flatten_amp,E,countries)
		vul_sort = list(temp)
		amp_sort = copy.deepcopy(vul_sort)
		E_sort = copy.deepcopy(vul_sort)
		vul_sort.sort(key=lambda tup: tup[0])
		amp_sort.sort(key=lambda tup: tup[1])
		E_sort.sort(key=lambda tup: tup[2])
		x_val = list()
		y_val = list()
		z_val = list()
		nodes = list()
		for i in range(len(E_sort)):
			for j in range(len(amp_sort)):
				for z in range(len(vul_sort)):
					if E_sort[i][3] == vul_sort[z][3] and E_sort[i][3] == amp_sort[j][3]:
						x_val.append(i+1)
						y_val.append(j+1)
						nodes.append(E_sort[i][3])
						z_val.append(z+1)
						break
		trace = go.Table(
		    header=dict(values=['Countries','Vulnerability','Amplification'],
		                fill = dict(color='#C2D4FF'),
		                align = ['left'] * 5),
		    cells=dict(values=[nodes,z_val, y_val],
		               fill = dict(color='#F5F8FF'),
		               align = ['left'] * 5))

		data = [trace] 
		plotly.offline.plot(data, filename = str(year) + '_table.html')
		print(year, " DONE")

def null_model(df):
	s_out = list()
	s_in = list()
	initial=0
	for col in df.columns:
		s_out.append(sum(df[col]))
		s_in.append(sum(df.loc[col]))
		for sub in df.columns:
			if(df.loc[sub,col]>0.0):
				initial+=1
	x_out = list()
	x_in = list()
	for i in range(len(s_out)):
		x_out.append(s_out[i]/sum(s_out))
		x_in.append(s_in[i]/sum(s_in))
	z = 1800000 # Between 1,700,000 and 1,800,000
	p = 0
	ad = list()
	degree = 0
	zeros = [0]*len(s_in)
	for i in range(len(s_out)):
		ad.append(zeros)
		for j in range(len(s_in)):
			if i!=j:
				p = (z*x_out[i]*x_in[j])/(1+(z*x_out[i]*x_in[j]))
				r =  random.uniform(0,1)
				print(r)
				if p>r:
					degree += 1
					ad[i][j] = 1
	# print("START")
	# while True:
	# 	temp_row = [0] * len(s_out)
	# 	temp_col = [0] * len(s_in)
	# 	print("START ROW")
	# 	for i in range(len(s_out)):
	# 		for j in range(len(s_in)):
	# 			if ad[i][j]!=0:
	# 				ad[i][j] = ((ad[i][j]/sum(ad[i]))*s_out[i])
	# 				temp_row[i] += ad[i][j]
	# 	print("START COL")
	# 	for i in range(len(s_in)):
	# 		for j in range(len(s_out)):
	# 			if ad[j][i]!=0:
	# 				ad[j][i] = ((ad[j][i]/sum(row[i] for row in ad))*s_in[i])
	# 				temp_col[i] += ad[j][i]
	# 	print("DOING ERROR")
	# 	error = convergence(temp_row,temp_col,s_out,s_in)
	# 	print(error)
	# 	if error<0.1:
	# 		break

def convergence(temp_row, temp_col,s_out,s_in):
	return rmse(np.array(temp_row)+np.array(temp_col), np.array(s_out)+np.array(s_in))


def country_aggregated_node_deletion():
	years = list(range(2000,2015))
	for year in years:
		df_2000 = pd.read_pickle('WIOD_Data/pickled_data/' + str(year) + '.pkl')
		countries = []
		countries_index = []
		i = 0
		for col in df_2000.columns:
			temp = col.split(',')
			if not(temp[len(temp)-1].lstrip() in  countries):
				countries.append(temp[len(temp)-1].lstrip())
				countries_index.append(i)
			i+=1
		countries_index.append(len(df_2000.columns))
		alpha, beta, one, m, M, E = model_intialization(df_2000)
		amplification = list()
		vulnerability = list()
		vulnerability = [0 for _ in range(len(countries_index)-1)]
		size = list()
		size = [0 for _ in range(len(countries_index)-1)]
		for x in range(len(countries_index)-1):
			M_next = copy.deepcopy(M)
			new_m = copy.deepcopy(m)
			new_alpha = copy.deepcopy(alpha)
			for i in range(countries_index[x], countries_index[x+1]):
				M_next[i] = [0 for _ in range(len(M_next[i]))]
				new_m[i] = [0 for _ in range(len(new_m[i]))]
				new_alpha[i] = 0
				for j in range(len(M_next)):
					M_next[j][i] = 0
					new_m[j][i] = 0
			size[x] += E[i]
			E_prev = E
			new_alpha_diag = np.diag(new_alpha)
			E_initial_shock = np.matmul(np.matmul(new_alpha_diag, np.transpose(M_next)),one) + beta
			E_initial_loss = E_initial_shock - E
			if(sum(E_initial_loss) != 0):
				for j in range(1, 100):
					E_next = np.matmul(np.matmul(new_alpha_diag, np.transpose(M_next)),one) + beta
					M_next = np.matmul(np.diag(E_next), new_m)
					rms = rmse(np.array(E_next), np.array(E_prev))
					if rms <= 0.1:
						break
					E_prev = E_next
				expenditure_difference = E_next - E
				for k in range(len(vulnerability)):
					if(k!=x):
						total_ex = 0
						initial_ex = 0
						for l in range(countries_index[k],countries_index[k+1]):
							total_ex += E_next[l]
							initial_ex += E[l]
						if(initial_ex!=0):
							vulnerability[k] += 1-total_ex/initial_ex
				amplification.append(sum(expenditure_difference)/sum(E_initial_loss))
			else:
				amplification.append(0)
		for j in range(len(vulnerability)):
			vulnerability[j] /= len(vulnerability)
		size = np.array(size)
		size = np.sqrt(np.array(size)/np.min(size[np.nonzero(size)]))
		plt.scatter(amplification, vulnerability,size)
		plt.title('Amplification vs Vulnerability')
		plt.xlabel('Vulnerability')
		plt.ylabel('Amplification')
		plt.savefig(str(year) + '_country_agg_ampvsvul_size.png')
		plt.clf()

if __name__ == "__main__":
	df_2000 = pd.read_pickle('WIOD_Data/pickled_data/2000.pkl')
	null_model(df_2000)
	# years = list(range(2000,2015))
	# for year in years:
	# 	df_2000 = pd.read_pickle('WIOD_Data/pickled_data/' + str(year) + '.pkl')
	# 	G = nx.DiGraph()
	# 	edges = list()
	# 	for row in df_2000.columns:
	# 		for col in df_2000.columns:
	# 			if col!=row and df_2000.loc[row,col]!=0.0:
	# 				edges.append((row,col,df_2000.loc[row,col]))
	# 	G.add_weighted_edges_from(edges)
	# 	nx.write_gpickle(G,str(year) + "_graph.gpickle")
	# 	print("FINISHED ", str(year))

	# G = nx.read_gpickle('2000_graph.gpickle')
	# nodes_centrality = nx.degree_centrality(G) # can do for in-degree and out-degree respectively and at the same time there is a method for betweeness centrality
	# r = nx.degree_assortativity_coefficient(G)
	# print(r)
	# total_degree_sequence = sorted([d for n, d in G.degree()], reverse=True) 
	# total_degreeCount = collections.Counter(total_degree_sequence)
	# total_deg, total_cnt = zip(*total_degreeCount.items())
	# in_degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True) 
	# in_degreeCount = collections.Counter(in_degree_sequence)
	# in_deg, in_cnt = zip(*in_degreeCount.items())
	# out_degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True) 
	# out_degreeCount = collections.Counter(out_degree_sequence)
	# out_deg, out_cnt = zip(*out_degreeCount.items())
	# fig, (ax1,ax2,ax3) = plt.subplots(3,1)
	# plt.tight_layout()
	# ax1.bar(total_deg, total_cnt, color='r')
	# ax1.set_title("Degree Distribution")
	# ax1.set_ylabel("Number of nodes")
	# ax1.set_xlabel("Total Degree")
	# ax2.bar(in_deg, in_cnt, color='b')
	# ax2.set_ylabel("Number of nodes")
	# ax2.set_xlabel("In Degree")
	# ax3.bar(out_deg, out_cnt, color='g')
	# ax3.set_ylabel("Number of nodes")
	# ax3.set_xlabel("Out Degree")
	# plt.show()

	# alpha, beta, one, m, M, E = model_intialization(df_2000)
	# results = node_deformation_shock((alpha, beta, one, m, M, E,0,9,0.4,0.3))
	# 	# start_node_deletion(df_2000) #average vulnerability
	# 	alpha, beta, one, m, M, E = model_intialization(df)
	# 	results = node_deletion_shock((alpha, beta, one, m, M, E,0,len(E)-1))
	# 	# with open('results_amp_vul.pickle', 'wb') as f:
	# 	# 	pickle.dump(results, f)

	# 	# results = pickle.load(open("results_amp_vul.pickle", "rb"))
	# 	size = np.array(E)/np.min(E[np.nonzero(E)])
	# 	amplification = list()
	# 	vulnerability = list()
	# 	# for tup in results:
	# 	# 	amplification.append(tup[0])
	# 	# 	vulnerability.append(tup[1])
	# 	# flatten_amp = [item for sublist in amplification for item in sublist]
	# 	# flatten_vul = [0 for _ in range(len(E))]
	# 	# for i in vulnerability:
	# 	# 	for j in range(len(flatten_vul)):
	# 	# 		flatten_vul[j] += i[j]
	# 	flatten_amp = results[0]
	# 	flatten_vul = results[1]
	# 	for i in range(len(flatten_vul)):
	# 		flatten_vul[i] /= len(E)
	# 	# plt.scatter(E, flatten_vul)
	# 	# plt.title('Vulnerability vs Expenditure')
	# 	# plt.xlabel('Expenditure')
	# 	# plt.ylabel('Vulnerability')
	# 	# plt.savefig(str(year) + '_agg_evsvul.png')
	# 	# plt.clf()
	# 	temp = zip(flatten_vul, flatten_amp,E,countries)
	# 	vul_sort = list(temp)
	# 	amp_sort = copy.deepcopy(vul_sort)
	# 	E_sort = copy.deepcopy(vul_sort)
	# 	vul_sort.sort(key=lambda tup: tup[0])
	# 	amp_sort.sort(key=lambda tup: tup[1])
	# 	E_sort.sort(key=lambda tup: tup[2])
	# 	x_val = list()
	# 	y_val = list()
	# 	z_val = list()
	# 	nodes = list()
	# 	for i in range(len(E_sort)):
	# 		for j in range(len(amp_sort)):
	# 			for z in range(len(vul_sort)):
	# 				if E_sort[i][3] == vul_sort[z][3] and E_sort[i][3] == amp_sort[j][3]:
	# 					x_val.append(i+1)
	# 					y_val.append(j+1)
	# 					nodes.append(E_sort[i][3])
	# 					z_val.append(z+1)
	# 					break
	# plt.scatter(flatten_vul[0:500], flatten_amp,s=size[0:500])
	# plt.title('Network robustness')
	# plt.xlabel('Vulnerability')
	# plt.ylabel('Amplification')
	# y = df_2000.columns
	# temp = zip(flatten_vul, flatten_amp,y)
	# vul_sort = list(temp)
	# amp_sort = copy.deepcopy(vul_sort)
	# vul_sort.sort(key=lambda tup: tup[0])
	# amp_sort.sort(key=lambda tup: tup[1])
	# x_val = list()
	# y_val = list()
	# nodes = list()
	# for i in range(len(vul_sort)):
	# 	for j in range(len(amp_sort)):
	# 		if vul_sort[i][2] == amp_sort[j][2]:
	# 			x_val.append(i+1)
	# 			y_val.append(j+1)
	# 			nodes.append(vul_sort[i][2])
	# 			break
	# plt.xticks(x_val, nodes)
	# plt.scatter(x_val,y_val,size)
	# plt.show()
