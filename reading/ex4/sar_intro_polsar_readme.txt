Polarimetry introduction:


0.Data:

Alling Data set of 2000 ('i_1206_xy.dat'), incidence angle ('i_1206_rla.dat') of scene and land use code of subset ('legend_alling.jpg', 'landuse_alling.jpg')


1.Literature:

Dissertation: Chapter 5 of Irena’s Dissertation
Book: Polarimetric RADAR imaging: From basics to applications (Lee, Pottier) – chapter 2-3 and chapter 6-7
Paper: A review of target decomposition theorems in RADAR polarimetry (Cloude & Pottier)
Paper: An entropy based classification scheme for land applications of polarimetric SAR (Cloude & Pottier)
Paper: Inversion of surface parameters from polarimetric SAR (Hajnsek & Pottier & Cloude)
Paper: A Four-Component Decomposition of PolSAR images based on the coherency matrix (Yamaguchi & Yajima & Yamada)

Futher literature:
Paper: Potential of estimating soil moisture under vegetation cover by means of PolSAR (Hajnsek & Jagdhuber & Schön & Papathanassiou)
Conference-Paper: Soil moisture estimation under Vegetation applying polarimetric decomposition techniques (Jagdhuber & Hajnsek & Papathanassiou)


2.Tasks:

Visualize power (absolute-squared) of [S]-Matrix Elements (Shh, Svv, Sxx) in dB (coloured). Take care: Sigma^0=Power of S-matrix-element*(sin(incidence))/1000000. Plus calculate the histograms for everything.
Calculate polarimetric coherences (HH-VV, HH-XX, VV-XX, LL-RR) and visualize the absolute and the phase of the coherence in black and white. Plus calculate the histograms for everything. 
Calculate the Covariance Matrix [C] and visualize the elements C1, C22, C33 as powers and the elements C13, C23, C12 as powers and their phases. Plus calculate the histograms for everything.
Calculate the Coherency Matrix [T] and visualize the elements T11, T22, T33 as powers and the elements T13, T23, T12 as powers and their phases. Plus calculate the histograms for everything.
Calculate the Total power (=span) of the [S]-, the [T]- and the [C]-matrix, visualize the images and plot all three histograms in one plot. What is the outcome?
Calculate the eigenvalues and eigenvectors of the [T]-matrix in an analytical way and with the built-in IDL routine ‘LA_Eigenql’. Compare both solutions. Differences between the solutions? Visualize the eigenvalues and compute the histograms? Do you notice anything, if you compare them?
Calculate the entropy (H), alpha angle (alpha), dominant alpha angle (alpha from dominant eigenvalue=alpha1) and anisotropy (A), Visualize in colour and plot the histograms.
Visualize entropy-alpha in a 2D-histogram plot and compare with the classification published in 4. Plus visualize H-A in a 2D-histogram plot.
Calculate X-Bragg model of  5. and show the modelled results for entropy and alpha in a 2D-entropy-alpha-plane.
Invert the calculated entropy and alpha layers from the data with the X-Bragg modelled entropy and alpha values for soil moisture and a roughness angle. Visualize the soil moisture and the roughness angle. Plus calculate the histograms for both.
Calculate a model-based decomposition on the [T]-matrix using 6., calculate the powers and visualize the powers. Is there a problem with this decomposition?


Please ask for help, if you are not advancing after half a day. You are not alone in this adventure. Please use the help-function of IDL for programming details. Feel free to ask for additional papers.

