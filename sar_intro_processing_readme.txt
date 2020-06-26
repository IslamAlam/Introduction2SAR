SAR/IDL introduction


Literature: 

Read IDL basics (idl_basics.pdf, chapters 1-7), or at least some of it, 
Cumming chapters (1-3, maybe 4-5), 
http://epsilon.nought.de/ (especially Synthetic Aperture Radar - Basic Concepts and Image Formation)
Alaska SAR Facility Processing User guide (alaska_sarprocessing.pdf).  


Data: 

2-D Simulation: chirp_2d_test.dat, chirp_2d_test_constants.txt
ERS: ers_raw_demo.dat, ers_constants.txt 
Airborne: rdemo040689_cmp.dat, rdemo_constants.txt


Tasks:

1. Exercise '1-D COMPRESSION': Create a range chirp signal 2000 ms long with chirp length 500 ms centred at 0. Perform convolution in time and in frequency domain (results should be the same) for 1-D compression.
2. Exercise '2-D COMPRESSION': Compress a simulated point (Result: Fresnel rings appear for 2-D SINC-function)
3. Exercise 'READ SPACEBORNE SAR DATA COMPRESSION WITH ERS-DATA':  Compress data in range and azimuth ( ers_raw_demo.dat) using the respective processing constants. (Remember to supply student with the necessary constants!) 
4. Exercise 'MULTI-LOOKING OF DATA IN FREQUENCY DOMAIN WITH ERS-DATA': Perform Multi-looking in range and azimuth direction 
5. Exercise 'REAL AIRBORNE SAR DATA COMPRESSION': The data is already compressed in range. Compression in azimuth with and without adapted azimuth chirp.

FURTHER DETAILS OF THE EXERCISES:


1-D COMPRESSION

1-D range chirp: 
Create a range chirp signal 2000 ms long with chirp length 500 ms centred at 0, set chirp coefficient to 500,
plot real and imaginary components of chirp, zero-padded chirp, amplitude of compressed chirp 
(perform matched filtering both in time and frequency domain: results should be the same!)
(IDL functions that may help: conj, reverse, convol, findgen, complex, complexarr, fft, abs, plot, window, !P.multi)

1-D range chirp with Hamming: (objective: to reduce side lobes)
plot FFT of chirp and Hamming window in frequency domain to see what is being cut/removed
plot compressed chirp with and without Hamming and compare resolutions
(IDL functions that may help: hanning with alpha=0.54, oplot, legend.pro)
(IDL functions that may help with statistics using mean, max,min)


2-D COMPRESSION

SIMULATED POINT RESPONSE 
use information in chirp_2d_test_constants.txt and chirp_2d_test.dat to focus a 2-D chirp
.DAT format: two longs (range and azimuth dimensions) followed by complex values
(IDL functions that may help: openr, readu, close, free_lun, lonarr, complexarr, shade_surf, hanning with alpha=0.54)
-compress 2D pulse using range chirp and azimuth chirp
-compress 2D pulse with Hamming in range and in azimuth
-compare both results


REAL SPACE BORNE SAR DATA COMPRESSION WITH ERS-DATA

USE ers_raw_demo.dat and ers_constants.txt to compress data in range and azimuth
-Plot data after range compression 
-Plot data after azimuth compression
-Plot data after azimuth/range compression with Hamming in azimtuh/ in range
(tip: you might want to scale the amplitudes after Hamming to conserve power)
(IDL functions that may help: tv, bytscl, congrid, hanning with alpha=0.54, shade_surf)


MULTI-LOOKING OF DATA IN FREQUENCY DOMAIN WITH ERS-DATA

-USE ers_raw_demo.dat and ers_constants.txt to compress data in range and azimuth 
  but taking only the half of the bandwidth for the range and the azimuth chirp 
  to create 4 looks 
-Visualize the result like Res=sqrt(|L1|²+|L2|²+|L3|²+|L4|²) with multi-looking for both chirps and compare with (original) single-look data.


REAL AIRBORNE SAR DATA COMPRESSION 

USE rdemo040689_cmp.dat and rdemo_constants.txt (data ALREADY range compressed)
-Plot data after azimuth compression with azimuth chirp from ERS data
-Plot data after azimuth compression with range-adapted azimuth chirp
-Plot data after azimuth compression  with range-adapted azimuth chirp AND Hamming
-What are the differences between the different compressions?

