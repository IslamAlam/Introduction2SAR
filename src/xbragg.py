import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import signal
import scipy
import pandas as pd
import seaborn as sns;


class XBragg():
    """| 1 2 3 | -> T_Matrix([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
       | 4 5 6 |
       | 7 8 9 |
    """
    def __init__(self, theta_inc, Epsilon_r, Beta1):
        #self.theta_inc = theta_inc
        #self.Epsilon_r = Epsilon_r 
        #self.Beta1 = Beta1 
        self.H, self.A, self.alpha_mean = self.XBraggModel(theta_inc, Epsilon_r, Beta1)
        
        #for ix,iy in np.ndindex(T_Matrix.shape[0:2]):
            #k_vec_temp = k_vector[ix, iy, :].reshape(np.sqrt(T_shape), np.sqrt(T_shape))
        #    T_mat = T_Matrix[ix, iy, :, :]
        #    w_temp, v_temp = self.EigenDecomposition(T_mat)
        #    self.w[ix, iy, :], self.v[ix, iy, :, :] = w_temp, v_temp
        #print(self.elems)
        #print(self.w)
    
    def XBraggModel(self, theta_inc, Epsilon_r, Beta1):
        """
        
        """
        self.T_XBragg = self.XBraggTMatrix(theta_inc, Epsilon_r, Beta1)
        
        # calculate eigenvalues and vectors of T Matrix
        self.w, self.v =  self.EigenDecomposition(self.T_XBragg)
        
        # calculate pseudo-probabilities p_i 
        self.p = self.PseudoProbabilities(self.w.real)
        
        # calculate Entropy
        self.H = self.Entropy(self.p)
        
        # calculate Anisotropy 
        self.A = self.Anisotropy(self.p)

        # calculate alpha 
        self.alpha, self.alpha_mean = self.AlphaAngles(self.v, self.p)

        return self.H, self.A, self.alpha_mean

        
    def XBraggTMatrix(self, theta_inc, Epsilon_r, Beta1):
        """
        Rs =
                 ______________         
                ╱         2             
            - ╲╱  εᵣ - sin (θ)  + cos(θ)
            ────────────────────────────
                ______________          
               ╱         2              
             ╲╱  εᵣ - sin (θ)  + cos(θ) 

        Rp =


            ⎡                 C₂⋅sin(2⋅β₁)                       ⎤
            ⎢     C₁          ────────────             0         ⎥
            ⎢                     2⋅β₁                           ⎥
            ⎢                                                    ⎥
            ⎢          __                                        ⎥
        T =  ⎢sin(2⋅β₁)⋅C₂     ⎛    sin(4⋅β₁)⎞                    ⎥
            ⎢────────────  C₃⋅⎜1 + ─────────⎟          0         ⎥
            ⎢    2⋅β₁         ⎝       4⋅β₁  ⎠                    ⎥
            ⎢                                                    ⎥
            ⎢                                     ⎛    sin(4⋅β₁)⎞⎥
            ⎢     0                0           C₃⋅⎜1 - ─────────⎟⎥
            ⎣                                     ⎝       4⋅β₁  ⎠⎦
        
        """
        N_Epsilon_r = len(Epsilon_r)
        N_Beta1 = len(Beta1)
        Rs = np.zeros(N_Epsilon_r, dtype=complex)
        Rp = np.zeros(N_Epsilon_r, dtype=complex)

        Rs = (-np.sqrt(Epsilon_r - np.sin(theta_inc)**2) + np.cos(theta_inc))/(np.sqrt(Epsilon_r - np.sin(theta_inc)**2) + np.cos(theta_inc))
        Rp = (Epsilon_r - 1)*(-Epsilon_r*(np.sin(theta_inc)**2 + 1) + np.sin(theta_inc)**2)/(Epsilon_r*np.cos(theta_inc) + np.sqrt(Epsilon_r - np.sin(theta_inc)**2))**2

        C_1 = np.abs(Rs + Rp)**2
        C_2 = (Rs + Rp)*(np.conjugate(Rs) - np.conjugate(Rp))
        C_3 = 1/2*np.abs(Rs - Rp)**2

        X_C_1, Y_beta = np.meshgrid(C_1, Beta1)
        X_C_2, Y_beta = np.meshgrid(C_2, Beta1)
        X_C_3, Y_beta = np.meshgrid(C_3, Beta1)


        T_XBragg = np.zeros(( N_Beta1, N_Epsilon_r, 3, 3), dtype=complex)

        T_XBragg[:, :, 0, 0] = X_C_1
        T_XBragg[:, :, 0, 1] = X_C_2 * np.sinc(2*Y_beta/np.pi)
        T_XBragg[:, :, 1, 0] = np.conj(X_C_2) * np.sinc(2*Y_beta/np.pi)
        T_XBragg[:, :, 1, 1] = X_C_3 * ( 1 + np.sinc(4*Y_beta/np.pi)) # 
        T_XBragg[:, :, 2, 2] = X_C_3 * ( 1 - np.sinc(4*Y_beta/np.pi)) #

        return T_XBragg
        

    def EigenDecomposition(self, T_Matrix):
        """
        W -> Eigenvlaues λ0 < λ1 < λ2
        V -> Eigenvectors
        """
        w = np.zeros((T_Matrix.shape[0], T_Matrix.shape[1], T_Matrix.shape[2]), dtype=float)
        v = np.zeros((T_Matrix.shape[0], T_Matrix.shape[1], T_Matrix.shape[2], T_Matrix.shape[3]), dtype=complex)
        
        for ix,iy in np.ndindex(T_Matrix.shape[0:2]):
            T_mat = T_Matrix[ix, iy, :, :]
            w[ix, iy, :], v[ix, iy, :, :]  = np.linalg.eigh(T_mat)            

        return w.real, v
    
    def PseudoProbabilities(self, w):
        p = np.zeros((w.shape), dtype=float)
        for col in range(w.shape[2]):
            p[:,:,col] = w[:,:,col] / np.sum(w, axis=2)
        return p
        
    def Entropy(self, p):
        H = -np.sum(p * np.log(p) / np.log(p.shape[2]), axis=2)
        return H
    
    def Anisotropy(self, w):
        A = (w[:,:,1] - w[:,:,0] ) / ( w[:,:,0] + w[:,:,1] ) # (lamda1 - lamda 2) / (lamda1 + lamda2)
        return A
    
    def AlphaAngles(self, v, p):
        alpha = np.zeros_like(p, dtype=np.float)
        alpha[:,:,0] = np.arccos(np.abs(v[:,:,0,0]))
        alpha[:,:,1] = np.arccos(np.abs(v[:,:,0,1]))
        alpha[:,:,2] = np.arccos(np.abs(v[:,:,0,2]))
        alpha_mean = (alpha[:,:,0]*p[:,:,0])+(alpha[:,:,1]*p[:,:,1])+(alpha[:,:,2]*p[:,:,2])

        return alpha, alpha_mean