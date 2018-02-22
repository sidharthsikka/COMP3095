import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

df_2000 = pd.read_pickle('WIOD_Data/pickled_data/2000.pkl')

# Delete any self pointing edges
for row in df_2000.iterrows():
    index, data = row
    for col in df_2000.columns:
        if index == col:
            df_2000.loc[index,col] = 0

# Delete any country where there is no trade
    for col in df_2000.columns:
        if df_2000[col].sum() == 0 and sum(df_2000.loc[col].tolist()) == 0:
            df_2000.drop(col, axis=1)
            df_2000.drop(col, axis=0)

G = nx.DiGraph()
for i in df_2000.columns.values:
    G.add_node(i)

weight_list = []

for i in df_2000.columns.values:
    for j in df_2000.columns.values:
        if i != j and df_2000.loc[i,j]>0:
            weight_list.append((i,j,df_2000.loc[i,j]))

G.add_weighted_edges_from(weight_list)
# nx.draw_spring(G)
# print("DRAWN")
# plt.savefig('2000_SPRING.png')


#Model
in_strength = [] #in strength of a node which is calculated for all the money coming into it
for row in df_2000.iterrows():
    index, data = row
    in_strength.append(sum(data.tolist()))
out_strength =[] #out strength of a node which is calculated for all the money going out of a node
for col in df_2000.columns:
    out_strength.append(df_2000[col].sum())
alpha = [] #combines i and o to give a propensity to spend for a sector by putting o/i if i>o else 1
for i in range(len(df_2000.columns)):
    if(in_strength[i] > out_strength[i]):
        if out_strength[i] != 0:
            alpha.append(in_strength[i]/out_strength[i])
        else:
            alpha.append((in_strength[i]))
    else:
        alpha.append(1)
beta = [] #again combines i and o to give a borrowing capacity of a company by putting o-i if o>i else 0
for i in range(len(df_2000.columns)):
    if(out_strength[i] > in_strength[i]):
        beta.append(out_strength[i] - in_strength[i])
    else:
        beta.append(0)
m = [] #for each connection i-j we normalize it by dividing the money going out to j by the total money going out
y = df_2000.columns
for i in range(len(out_strength)):
    m.append(df_2000[y[i]]/out_strength[i])
M = np.matmul(np.diag(out_strength),m) #diagonal matrix of o multiply m
one = [1 for _ in range(len(df_2000.columns))]
E = np.matmul(np.matmul(np.diag(np.transpose(alpha)), np.transpose(M)),np.transpose(one)) + beta
amplification_post = [] # for each shock this stores the amplification score for plotting
amplification_pre = []
for i in range(len(df_2000.columns)):
    M_T = []
    M_T.append(M)
    for j in range(len(M)):
        if i==j:
            M[j] = 0
    M_1 = [] #Depending on the kind of shock you have to populate this and pass it into M_T, calculate E1 and pass it to E_T and then iterate with the below for loop
    E_T = []
    E_T.append(E)
    for i in range(10):
        M_T.append(np.matmul(np.diag(E_T[i]),m))
        E_T.append(np.matmul(np.matmul(np.diag(np.transpose(alpha)), np.transpose(M)),np.transpose(one)) + beta)
        rms = sqrt(mean_squared_error(E_T[i], E_T[i+1]))
        if rms == 0:
            print("CONVERGED")

    expenditure_difference_post = E_T[len(E_T)] - E_T[1] #in comparison after initial shock has been applied
    expenditure_difference_pre = E_T[len(E_T)] - E_T[0] #in comparison before initial shock has been applied
    amplification_post.append(sum(expenditure_difference_post))
    amplification_pre.append(sum(expenditure_difference_pre))

# Plotting the amplification of removing each node
plt.subplot(2, 1, 1)
plt.plot(df_2000.columns, amplification_post, 'o-')
plt.title('Amplification VS Node Deletion')
plt.ylabel('Amplification')

plt.subplot(2, 1, 2)
plt.plot(df_2000.columns, amplification_pre, '.-')
plt.xlabel('Industries')
plt.ylabel('Amplification')

plt.show()
