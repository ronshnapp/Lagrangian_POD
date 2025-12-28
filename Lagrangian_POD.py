# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Sun Dec 28 16:59:25 2025

@author: ron
"""

import numpy as np
import pandas as pd
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import tqdm






# ===========================================
# THE ACTUAL CODE FOR CALCULATION THE L-POD
# ===========================================


class LPOD(object):
    
    def __init__(self, data):
        '''
        A class to calculate Lagrangian POD stuff.
        Data is a list of (n X 11) arrays representing
        trajectories with the same length n.
        '''
        self.data = data
        self.nt = len(self.data[0])
        self.ntr = len(data)
        
        for i in range(self.ntr):
            if len(self.data[i]) != self.nt:
                raise ValueError('all trajs must be of same length')
    
    
    def calculate_POD(self):
        '''
        Calculates the modes and time coefficients matrixes
        '''
        # organize the data matrix
        u = []
        for tr in self.data:
            u.append(tr[:,4])
            u.append(tr[:,5])
            u.append(tr[:,6])

        u = np.array(u).T
        
        # remove the mean 
        self.U = np.mean(u, axis=0)
        u_fluct = u - self.U

        # calculate the covariance matrix
        cov = (u_fluct.T @ u_fluct) / (self.nt-1)

        # calculate egen values and eigen vectors
        # (v[:,i] is the eigen vector with eigen value l[i])
        l, v = np.linalg.eig(cov)

        # sort l and v according to l
        d = list(zip(*sorted(zip(l, list(v.T)), key=lambda x: x[0], reverse=True)))
        l = np.array(d[0])
        v = np.array(d[1]).T
        
        self.l = l
        self.v = v

        # get time coefficients
        a = u_fluct @ v
        a.shape, v.shape
        
        self.a = a
        
        self.trajectory_modes = np.array([np.reshape(v[:,i], (len(self.data), 3)) for i in range(v.shape[1])])
        self.mean_traj_vel = np.reshape(self.U, (len(subset), 3))
           

    def get_traj_from_velocities(self, velocity, orig_traj):
        '''
        Given a (n X 3) array containing n velocity samples,
        this function will return a MyPTV-like trajectory 
        array. The format is assumes as a smoothed trajectory,
        namely a trajectory is a (n X 11) numpy array.

        We estimate the position by integrating the velocities.
        we estimate the acceleration by differentiating them.
        the original trajectory is used to get the time, initial
        position and trajectory id values.
        '''
        new_traj = orig_traj.copy()
        tm = new_traj[:,-1]

        dx = [simpson(velocity[:i,0], tm[:i]-tm[0]) for i in range(1,len(orig_traj))]
        dx.append(dx[-1] + velocity[-2,0]*(tm[1]-tm[0]))
        new_traj[:,1] = np.array(dx) + orig_traj[0,1]

        dy = [simpson(velocity[:i,1], tm[:i]-tm[0]) for i in range(1,len(orig_traj))]
        dy.append(dy[-1] + velocity[-2,1]*(tm[1]-tm[0]))
        new_traj[:,2] = np.array(dy) + orig_traj[0,2]

        dz = [simpson(velocity[:i,2], tm[:i]-tm[0]) for i in range(1,len(orig_traj))]
        dz.append(dz[-1] + velocity[-2,2]*(tm[1]-tm[0]))
        new_traj[:,3] = np.array(dz) + orig_traj[0,3]

        new_traj[:,4] = velocity[:,0]
        new_traj[:,5] = velocity[:,1]
        new_traj[:,6] = velocity[:,2]

        new_traj[:,7] = np.gradient(velocity[:,0])
        new_traj[:,8] = np.gradient(velocity[:,1])
        new_traj[:,9] = np.gradient(velocity[:,2])

        return new_traj
    
    
    def reconstruct_traj_K(self, K, ntr):
        '''
        returns the low order reconstruction of trajectory number ntr using 
        the first K modes.
        
        K - an integer
        ntr - an integer
        '''
        # reconstruct the trajectory velocity  
        traj_vel = np.dot(self.a[:,:K], self.trajectory_modes[:K, ntr, :]) + self.mean_traj_vel[ntr]
        
        # integrate the trajectory data
        tr_k = self.get_traj_from_velocities(np.real(traj_vel), self.data[ntr])
        return tr_k





# ===========================================
# EXAMPLE OF USING THE L-POD ON REAL DATA
# ===========================================


#%% 
# 0.0) Loading trajectory data (designed using MyPTV file formats)

trajs = []
path = './data_sample'
fnames = [path]

for fname in fnames:
    data = pd.read_csv(fname, header=None, sep='\t')
    data = data[data[0]!=-1]
    trajs = trajs + [np.array(g) for k,g in data.groupby(0)]


# 0.1) Taking a subset of the data with trajectories of length L

L = 150
subset = [tr[:L] for tr in trajs if len(tr)>=L]
print(len(subset))




#%%
# 1) calculating the modes:
pod = LPOD(subset)
pod.calculate_POD()




#%%
# 2) Plotting the modes
cmap = plt.get_cmap('viridis')

K_lst = [1,2,3,4,5,6,7,8,9,10]

fig, ax = plt.subplots(len(K_lst)//2, 2, sharey=True, sharex=True)


for e,K in enumerate(K_lst):
    ax[e%5][e//5].plot(np.real(np.dot(pod.a[:,K-1:K], 1)), color=cmap((K)/len(K_lst)))
    ax[e%5][e//5].plot([0,150], [0,0], 'k-', lw=0.5)
    ax[e%5][e//5].text(135, 2.7,'K=%d'%K)
plt.tight_layout()






#%%
# 3) Show the decomposition on a single trajectory


cmap = plt.get_cmap('viridis')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ntr = 2

tr = tr_k = pod.reconstruct_traj_K(-1, ntr)
ax.plot(tr[:,1], tr[:,3], tr[:,2], 'k--')

for e, K in enumerate([0,1,2,3,4,5]):
    tr_k = pod.reconstruct_traj_K(K, ntr)
    ax.plot(tr_k[:,1], tr_k[:,3], tr_k[:,2], '-', color=cmap(e/5))
    
    
ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

plt.tight_layout()




#%% 
# 4) Mode energies


print(pod.l.shape)

tot = np.real(np.sum(pod.l))
fig, ax = plt.subplots()
ax.semilogy(np.real(pod.l[:150]/tot), 'o-', ms=3, lw=0.8)
ax.set_ylim(bottom=1e-8)





#%%
# 5) Showing the modes as a 2D color map


fig, ax = plt.subplots()

ax.imshow(np.real(pod.a)[:,:50], cmap='bwr')
ax.set_aspect(0.3)
ax.grid()




#%%
# 6) Making a cottonball plot of trajectories



K = -1    # sum of all modes

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for ntr in tqdm.tqdm(range(len(subset))):
    # reconstruct the trajectory velocity  
    traj_vel = np.dot(pod.a[:,:K], pod.trajectory_modes[:K, ntr, :])

    # integrate the trajectory data
    tr_k = pod.get_traj_from_velocities(np.real(traj_vel), pod.data[ntr])

    # plot the traj
    U = np.mean(np.sum(tr_k[:,4:7]**2, axis=1)**0.5)
    ax.plot(tr_k[:,1]-tr_k[0,1], tr_k[:,3]-tr_k[0,3], tr_k[:,2]-tr_k[0,2], color=cmap(U/0.3))
        
        
ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

plt.tight_layout()





















