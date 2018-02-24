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

# Original Data is expressed in millions of dollars so this will bring them all to dollars
for col in df_2000.columns:
    df_2000[col] = df_2000[col] * 1000000

'''G = nx.DiGraph()
for i in df_2000.columns.values:
    G.add_node(i)

weight_list = []

for i in df_2000.columns.values:
    for j in df_2000.columns.values:
        if i != j and df_2000.loc[i,j]>0:
            weight_list.append((i,j,df_2000.loc[i,j]))

G.add_weighted_edges_from(weight_list)'''
# nx.draw_spring(G)
# print("DRAWN")
# plt.savefig('2000_SPRING.png')


#Model

#in strength of a node which is calculated for all the money coming into it
in_strength = []
for row in df_2000.iterrows():
    index, data = row
    in_strength.append(sum(data.tolist()))

#out strength of a node which is calculated for all the money going out of a node
out_strength = []
for col in df_2000.columns:
    out_strength.append(df_2000[col].sum())

#combines i and o to give a propensity to spend for a sector by putting o/i if i>=o else 1
alpha = []
for i in range(len(df_2000.columns)):
    if in_strength[i] >= out_strength[i]:
        if in_strength[i] != 0: #ASSUMPTION
            alpha.append(out_strength[i]/in_strength[i])
        else:
            alpha.append(out_strength[i])
    else:
        alpha.append(1)

#again combines i and o to give a borrowing capacity of a company by putting o-i if o>i else 0
beta = []
for i in range(len(df_2000.columns)):
    if(out_strength[i] > in_strength[i]):
        beta.append(out_strength[i] - in_strength[i])
    else:
        beta.append(0)

#for each connection i-j we normalize it by dividing the money going out to j by the total money going out
m = []
y = df_2000.columns
for i in range(len(out_strength)):
    if out_strength[i] != 0: #ASSUMPTION
        m.append((df_2000[y[i]]/out_strength[i]).tolist())
    else:
        m.append(df_2000[y[i]])

#INITIAL M & E
M = np.matmul(np.diag(out_strength),m)
one = [1 for _ in range(len(df_2000.columns))]
E = np.matmul(np.matmul(np.diag(np.transpose(alpha)), np.transpose(M)),np.transpose(one)) + np.transpose(beta)

amplification_post = [] # for each shock this stores the amplification score for plotting
amplification_pre = []
print("STARTING SHOCKS")
for i in range(len(df_2000.columns)):
    M_T = []
    M_T.append(M)
    E_T = []
    E_T.append(E)
    M_1 = M.copy()
    M_1[i] = 0
    for j in range(len(M_1)):
        M_1[j][i] = 0
    M_T.append(M_1)
    E_T.append(np.matmul(np.matmul(np.diag(np.transpose(alpha)), np.transpose(M_1)),np.transpose(one)) + np.transpose(beta))
    for j in range(1,10):
        M_T.append(np.matmul(np.diag(E_T[j]),m))
        E_T.append(np.matmul(np.matmul(np.diag(np.transpose(alpha)), np.transpose(M_T[j+1])),np.transpose(one)) + np.transpose(beta))
        print(sum(E_T[j + 1]))
        print(sum(E_T[j]))
        '''rms = sqrt(mean_squared_error(E_T[j+1], E_T[j]))
        if rms == 0:
            print("CONVERGED")
            break

    expenditure_difference_post = E_T[len(E_T)-1] - E_T[1] #in comparison after initial shock has been applied
    expenditure_difference_pre = E_T[len(E_T)-1] - E_T[0] #in comparison before initial shock has been applied
    amplification_post.append(sum(expenditure_difference_post))
    amplification_pre.append(sum(expenditure_difference_pre))'''
    print("SHOCK DONE")

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