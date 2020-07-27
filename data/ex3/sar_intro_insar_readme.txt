InSAR introduction

1. Reading: Kostas' thesis chapter 4, bamler98a_insar_review.pdf, and gatelli94_wavenumshift.pdf. 
2. Exercise: Process Mt. Etna i_sar data.  (coreg of 2 images, removal of flat earth phase, compute coherence, range filtering)
3. Read more about phase unwrapping (bamler98b_slopedistortion_phsunwrp.pdf and pritt94_LSE_FFT_phsunwrp.pdf). 
4. Exercise: Phase unwrap Etna data using least squares.


FURTHER DETAILS:

In this exercise you will apply the concepts from the papers to data from Mount Etna.  

A. COREG and FLAT EARTH PHASE
1. Read-in the two .dat files.  
2. Coregistration of 2 images  
 [HINT: perform 2-D cross-correlation (most efficient when performed in frequency domain!) to find range/azimuth shifts between image 1 and image 2.  Shift image 2 so that it aligns with image 1.]
3. Remove flat earth phase  [HINT: look for dominant frequency component (i.e. the fringe frequency) in interferogram and shift image 2]
4. Display image 1 and image 2 magnitudes.  Compute/display coherence and interferometric phase BEFORE and AFTER flat earth phase removal.  Try different window sizes for the coherence computation.  

B.  RANGE FILTERING
1. Read-in the two .dat files.
2. Coregistration of 2 images  
3. Perform range filtering (according to Gatelli94, i.e. preserve only overlapping bandwidth between master and slave).
4. Remove flat earth phase.
5. Compute coherence and interferometric phase (BEFORE and AFTER range filtering) and display (show colorbars).  Display histograms of coherences BEFORE and AFTER range filtering.

(new IDL functions that might be of use: FUNCTION - try to split each step above into a separate function for reusability)
(SQRT, SMOOTH, XYOUTS, LOADCT, HISTOGRAM)

Here is one way of making a color bar (try to understand the various parameters):
color_bar = Obj_New('COLORBAR', vertical=1, range=[minval,maxval], position = [0.96, 0.2, 1.3, 0.80], charsize=1, color=0)
color_bar->Draw

Color LUT (look-up-table) #13 and 39 are a nice rainbow of colors for display purposes.


