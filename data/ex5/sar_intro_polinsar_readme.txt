Pol-InSAR introduction:

0.Data:
sim_rvog_data.zip

1.Literature:
Cloude & Papathanassiou "Polarimetric SAR Interferometry",1998.
Papathanassiou & Cloude "Single-Baseline Polarimetric SAR Interferometry", 2001.
Cloude & Papathanassiou "Three-stage inversion process for polarimetric SAR interferometry", 2003.


2.Constants for the calculation:
;constants for sim_data_rvog
H = 3e3   ;(sensor height, m)
lambda = 0.24  ;(wavelength, L-band, m)
B_12 = -10  ;(horz baseline btw Im1&2, m)
B_13 = -20  ;(horz baseline btw Im1&3, m)
W = 100e6  ;(bandwidth in range, Hz)
grng_res = 0.5  ;(ground rng pixel spacing, m)
theta0 = 45*!DTOR ;(angle of incidence to img centre, assume constant for small area) (rads)
c = 3e8   ;(speed of light, m/s)
Rm = H/cos(theta0) ;broadside range (m)
alpha = 0  ;local slope 

3.Description of Exercise
The aim of the exercise is the determination of the forest height according to the 3 stage inversion paper from Shane Cloude and Kostas Papathanassiou, 2003. This requires the following steps:
 
1.read the complex data from the attachment. Plot amplitudes. Compute and plot coherence magnitudes and coherence phases for baseline 12 and baseline 13 (i.e. baseline12  = image 1 and image 2).

2.calculation of the Pauli components 

3.flat earth correction: Compute flat-earth phase analytically using the given constants and remove from slave pass.
i.e. Determine geometric phase difference between pass 1 and pass 2 and pass 1 and pass 3 assuming flat topography. 

4.calculation of hh,vv, xx, pauli1 (HH+VV) and pauli2 (HH-VV) coherences 

5.plot magnitude and phase of coherence for hh, vv, xx (both baselines) in a 2D-plot 

6.compare in a histogram the magnitude of the coherence of hh, vv and xx (both baselines) 

7.plot the magnitude and phase of coherence for hh, vv, xx, pauli1 and pauli2  (baseline 12) in the unit circle 

8.start 3-stage Pol-InSAR inversion for topographic phase, extinction, and tree height: Make a line fit (Linear fit to coherences. Plot complex unit circle and locations of 5 coherence points.  Fit line through the points.) HINT: use IDL LINFIT routine. To avoid infinite slopes one can also try shifting the points 90 degs. and re-compute LINFIT.  Take the solution with the lowest CHISQ error.

9.find intersection(s) between best-fit line and the unit circle and project the points onto the line. HINT: use math!  equation of a line and equation of a circle gives 2 eqns and two unknowns (coordinates of point of intersection)

10.determine the ground phase. HINT: Find Euclidean distance from each coherence to each of the two possible ground points.  Determine where the XX coherence (typically has a smaller ground contribution than the other coherences) lies in relation to the other coherences (= intersection point of line with circle that is further away from xx coherence). 

11.determine the ground phase (see 8 to 10) for all points 

12.show ground phases of all points in a histogram 

13.create and plot lookup table (LUT) for coherence of vegetation: 
Assume XX coherence (projected to best-fit line) has no ground contribution. Perform integral from eqn. 8 (Cloude&Papathanassiou 2003). Compute 2D LUT (look-up table) with different values of extinction (eg. vary from 0 to 2 dB/m) and height (eg. vary from 0 to 30 m)

14.calculate vegetation height and sigma (extinction) from LUT 

15.plot vegetation height histogram and 3D-plot (shade surf) 

Please ask for help, if you are not advancing after half a day. You are not alone in this adventure. Please use the help-function of IDL for programming details. Feel free to ask for additional papers.

