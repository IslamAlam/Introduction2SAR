import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import signal
import scipy
import pandas as pd
import seaborn as sns;



class PolSAR():
    def __init__(self, XX, XY, YX, YY):
        print("t")
        
        self.SMatrix = self.MonostaticBackscatteringMatrixS(XX, XY, YX, YY)
        
        self.lex_vector = self.lexicographic_scattering_vector(self.SMatrix)
        self.CMatrix = self.PolarimetricCovarianceCMatrix(omega_vector=self.lex_vector)

        self.pauli_vector = self.pauli_scattering_vector(self.SMatrix)
        self.TMatrix = self.PolarimetricCoherencyTMatrix(k_vector=self.pauli_vector)
        print(self.lex_vector.shape)
        
        
        
    def MonostaticBackscatteringMatrixS(self, HH, HV, VH, VV):
        """
        [ 𝑆 ]=[ 𝑆_𝐻𝐻 𝑆_𝑉𝐻 
              𝑆_𝐻𝑉 𝑆_𝑉𝑉  ]
        3.5.1 ................................................. pg 80
        """
        scat_matrix = np.zeros((HH.shape[0], HH.shape[1], 4), dtype=complex)
        scat_matrix[:,:,0] = HH
        scat_matrix[:,:,1] = HV
        scat_matrix[:,:,2] = VH
        scat_matrix[:,:,3] = VV
        return scat_matrix
    
    def lexicographic_scattering_vector(self, S_Matrix):
        """
        𝑘⃗_3𝐿 =[ 𝑆_𝐻𝐻, √2 𝑆_𝐻𝑉, 𝑆_𝑉𝑉 ]𝑇
        """
        if not np.array_equal(S_Matrix[:,:,1], S_Matrix[:,:,2]):
            '''
            3.2.2 BISTATIC SCATTERING CASE
            4-D Lexicographic feature vector
            '''
            print('4-D Lexicographic feature vector')
            lex_vector = np.zeros((S_Matrix.shape[0], S_Matrix.shape[1], 4), dtype=complex)
            lex_vector[:,:,0] = S_Matrix[:,:,0]
            lex_vector[:,:,1] = S_Matrix[:,:,1]
            lex_vector[:,:,2] = S_Matrix[:,:,2]
            lex_vector[:,:,3] = S_Matrix[:,:,3]

        else:
            '''
            3.2.3 MONOSTATIC BACKSCATTERING CASE
            3-D Lexicographic feature vector
            '''
            print('3-D Lexicographic feature vector')
            lex_vector = np.zeros((S_Matrix.shape[0], S_Matrix.shape[1], 3), dtype=complex)
            lex_vector[:,:,0] = S_Matrix[:,:,0]
            lex_vector[:,:,1] = np.sqrt(2) * S_Matrix[:,:,1]
            lex_vector[:,:,2] = S_Matrix[:,:,3]
        return lex_vector

    def pauli_scattering_vector(self, S_Matrix):    
        '''
        Pauli scattering vector
        𝑘⃗_3𝑃 = 1/√2 [ 𝑆_𝐻𝐻+𝑆_𝑉𝑉, 𝑆_𝐻𝐻−𝑆_𝑉𝑉, 2𝑆_𝐻𝑉 ]𝑇
        '''
        if not np.array_equal(S_Matrix[:,:,1], S_Matrix[:,:,2]):
            '''
            3.2.2 BISTATIC SCATTERING CASE
            4-D Pauli feature vector
            '''
            print('4-D Pauli feature vector')
            pauli_vector = np.zeros((S_Matrix.shape[0], S_Matrix.shape[1], 4), dtype=complex)
            pauli_vector[:,:,0] = S_Matrix[:,:,0] + S_Matrix[:,:,3]
            pauli_vector[:,:,1] = S_Matrix[:,:,0] - S_Matrix[:,:,3]
            pauli_vector[:,:,2] = S_Matrix[:,:,1] + S_Matrix[:,:,2]
            pauli_vector[:,:,3] = 1j*(S_Matrix[:,:,1] - S_Matrix[:,:,2])
            pauli_vector = pauli_vector/np.sqrt(2)
        else:
            '''
            3.2.3 MONOSTATIC BACKSCATTERING CASE
            3-D Pauli feature vector
            '''
            print('3-D Pauli feature vector')
            pauli_vector = np.zeros((S_Matrix.shape[0], S_Matrix.shape[1], 3), dtype=complex)
            pauli_vector[:,:,0] = S_Matrix[:,:,0] + S_Matrix[:,:,3]
            pauli_vector[:,:,1] = S_Matrix[:,:,0] - S_Matrix[:,:,3]
            pauli_vector[:,:,2] = 2*S_Matrix[:,:,1]
            pauli_vector = pauli_vector/np.sqrt(2)
        return pauli_vector


    def SigmaPowerMatrix():
        """
        Visualize power (absolute-squared) of [S]-Matrix Elements (Shh, Svv, Sxx) in dB (coloured). Take care: Sigma^0=Power of S-matrix-element*(sin(incidence))/1000000. Plus calculate the histograms for everything
        """
        
    def PolarimetricCoherencyTMatrix(self, k_vector):
        """
        Calculate the Coherency Matrix [T] and visualize the elements T11, T22, T33 as powers and the elements T13, T23, T12 as powers and their phases. Plus calculate the histograms for everything
        3.5.2 ..................................................... pg 83
        Polarimetric coherency matrix [T 3x3]
        
        """

        # k_vector = pauli_scattering_vector(S_Matrix)
        pol_coherency_matrix = np.zeros(
            (k_vector.shape[0], k_vector.shape[1], k_vector.shape[2], k_vector.shape[2]), dtype=complex)

        pol_mat_temp = np.zeros((k_vector.shape[2], k_vector.shape[2]), dtype=complex )
        
        k_vec_temp = np.zeros( (k_vector.shape[2], 1),  dtype=complex )
        
        for ix,iy in np.ndindex(k_vector.shape[0:2]):
            k_vec_temp = k_vector[ix, iy, :].reshape(-1,1)
            pol_mat_temp = np.dot(k_vec_temp, k_vec_temp.T.conjugate())
            pol_coherency_matrix[ix, iy, :, :] = pol_mat_temp

        n_window = 7
        mean_filter = np.ones((n_window, n_window))
        mean_filter /=sum(mean_filter)
        pol_coherency_matrix_filtered = np.zeros_like(pol_coherency_matrix)
        for ix,iy in np.ndindex(pol_coherency_matrix.shape[2:]):
            # print(i)
            pol_coherency_matrix_filtered[:, :, ix, iy] = scipy.signal.convolve(pol_coherency_matrix[:, :, ix, iy] , 
                                                                         mean_filter, mode='same')
        return pol_coherency_matrix, pol_coherency_matrix_filtered
    
    
    def PolarimetricCovarianceCMatrix(self, omega_vector):
        """
        Calculate the Covariance Matrix [C] and visualize the elements C1, C22, C33 as powers and the elements C13, C23, C12 as powers and their phases. Plus calculate the histograms for everything
        3.5.3 .................................................... 84
        Polarimetric covariance matrix [𝐶 3x3] or [C 4x4]
        """
        # omega_vector = lexicographic_scattering_vector(S_Matrix)
        pol_covariance_matrix = np.zeros(
            (omega_vector.shape[0], omega_vector.shape[1], omega_vector.shape[2], omega_vector.shape[2]), dtype=complex)

        pol_mat_temp = np.zeros((omega_vector.shape[2],omega_vector.shape[2]), dtype=complex )
        
        omega_vec_temp = np.zeros( (omega_vector.shape[2], 1),  dtype=complex )
        for ix,iy in np.ndindex(omega_vector.shape[0:2]):
            omega_vec_temp = omega_vector[ix, iy, :].reshape(-1, 1)
            pol_mat_temp = np.dot(omega_vec_temp, omega_vec_temp.T.conjugate())
            pol_covariance_matrix[ix, iy, :, :] = pol_mat_temp

        n_window = 7
        mean_filter = np.ones((n_window, n_window))
        mean_filter /=sum(mean_filter)
        pol_covariance_matrix_filtered = np.zeros_like(pol_covariance_matrix)
        for ix,iy  in np.ndindex(pol_covariance_matrix.shape[2:]):
            # print(i)
            pol_covariance_matrix_filtered[:,:,ix,iy] = scipy.signal.convolve(pol_covariance_matrix[:,:,ix,iy] , 
                                                                         mean_filter, mode='same')
        return pol_covariance_matrix, pol_covariance_matrix_filtered
    
    
    def PolarimetricKennaughKMatrix(self, ):
        """
        3.5.4 ...................................................... 84
        """
        

        
        
class EigenvectorDecomposition():
    """| 1 2 3 | -> T_Matrix([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
       | 4 5 6 |
       | 7 8 9 |
    """
    def __init__(self, T_Matrix):
        # self.elems = T_Matrix
        self.w, self.v = self.EigenDecomposition(T_Matrix)
        
        self.p = self.PseudoProbabilities(self.w.real)
        self.H = self.Entropy(self.p)
        self.A = self.Anisotropy(self.p)
        self.alpha, self.alpha_mean = self.AlphaAngles(self.v, self.p)
        
        #print(self.elems)
        #print(self.w)
    
    def EigenDecomposition(self, T_Matrix):
        """
        W -> Eigenvlaues w0 < w1 < w2
        V -> Eigenvectors
        """
        w = np.zeros((T_Matrix.shape[0], T_Matrix.shape[1], T_Matrix.shape[2]), dtype=float)
        v = np.zeros((T_Matrix.shape[0], T_Matrix.shape[1], T_Matrix.shape[2], T_Matrix.shape[3]), dtype=complex)
        
        for ix,iy in np.ndindex(T_Matrix.shape[0:2]):
            T_mat = T_Matrix[ix, iy, :, :]
            w[ix, iy, :], v[ix, iy, :]  = np.linalg.eigh(T_mat)            

        return w.real, v
    
    def PseudoProbabilities(self, w):
        p = np.zeros((w.shape), dtype=float)
        for col in range(w.shape[2]):
            print(col)
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
        print(v.shape, p.shape)
        alpha[:,:,0] = np.arccos(np.abs(v[:,:,0,0]))
        alpha[:,:,1] = np.arccos(np.abs(v[:,:,0,1]))
        alpha[:,:,2] = np.arccos(np.abs(v[:,:,0,2]))
        alpha_mean = (alpha[:,:,0]*p[:,:,0])+(alpha[:,:,1]*p[:,:,1])+(alpha[:,:,2]*p[:,:,2])

        return alpha, alpha_mean
