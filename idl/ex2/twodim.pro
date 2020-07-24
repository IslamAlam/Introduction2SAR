
; @G:\SAR_Tutorial_02_Exercises\sar\rect.pro
; @/media/ADATA_UFD/SAR_Tutorial_02_Exercises/sar/rect.pro
@/home/mans_is/mans_is/repos/Introduction2SAR/idl/ex2/rect.pro

PRO twodim


; --- SETTINGS FOR THE PLOTS
!p.background=255
!p.color=0
!p.thick=2
!p.charsize=2
!p.charthick=2
device,decompose=0,retain=2
loadct,0,/silent
; folder='G:\SAR_Tutorial_02_Exercises\sar\'
; folder='/media/ADATA_UFD/SAR_Tutorial_02_Exercises/sar/'
folder='/home/mans_is/mans_is/repos/Introduction2SAR/data/ex2/'
thisLetter = "154B
micro = '!4' + String(thisLetter) + '!X'

; --- DEFINE CONSTANTS
z = complex(0,1)
; closest range, (m)
R0 = 1000d
; sensor velocity (m/s) 		
v = 70 	  		
; antenna length (m)
L_antenna = 2	
; wavelength (m)	
lambda = 0.0566d	
; speed of light (m/s)
C = 3d8		
; bandwidth (Hz)	
Bw = 50d6		
; pulse width (s)
Tau = 5d-6		
; sampling frequency in range (Hz)
Fs_range = 100d6	
; pulse repetition frequency (Hz)
PRF = 400 		

; --- READ DATA
OPENR, 1, folder+'chirp_2d_test.dat', /XDR
dim = LONARR(2)
READU, 1, dim
data = COMPLEXARR(dim(0),dim(1))
READU, 1, data
CLOSE, 1
Nrg=dim(0)
Naz=dim(1)

; --- DERIVED QUANTITIES
; antenna beamwidth
thet = lambda/L_antenna 
; max illumination time
tmax = thet*R0/v  
; range chirp coefficient
K = Bw/Tau    
; azimuth chirp coefficient
KA = 2*v^2/(lambda*R0)  
; time axis in range
time = findgen(Nrg)/Fs_range  
; time axis in azimuth
time_a = findgen(Naz)/PRF  
; frequency axis in range
frec = findgen(Nrg)/(Nrg-1)*Fs_range-Fs_range/2
; frequency axis in azimuth
frec_a = findgen(Naz)/(Naz-1)*PRF-PRF/2

; --- DESIGN THE CHIRP TO COMPRESS SIGNAL IN TIME DOMAIN
; - compression in range
time = time-Nrg/Fs_range/2       ; shift time to symmetric array
env = rect(time/Tau)				
mfr = dcomplexarr(Nrg)				
mfr = env*exp(!pi*z*K*(time)^2)  ; filter in time domain to compress in range
; plots
window,0, xs=1450, ys=600, title='Processing along range'
!P.MULTI = [0,3,1]
; range - matched filter - phase
plot, time/(1e-6), atan(mfr,/ph), yrange=[-5,5],xrange=[-Nrg/Fs_range/2*1e6,Nrg/Fs_range/2*1e6], XTITLE = 'Time ['+micro+'s]', YTITLE = "Phase [rad]",TITLE='Matched filter, range - Ph.',xstyle=1,ystyle=1,background=255,color=0
; range - matched filter - frequency response
plot, frec/1e6, abs(shift(fft(mfr),Nrg/2)),yrange=[0,0.1],xrange=[-Fs_range/2/1e6,Fs_range/2/1e6],xstyle=1,ystyle=1,xtitle='Frequency [MHz]',ytitle='[-]',title='Matched filter, range - Response',background=255,color=0
; data compression
datacompr = COMPLEXARR(Nrg,Naz)
for i=0,Naz-1 do datacompr(*,i) = CONVOL(data(*,i),mfr,/edge_zero)	; apply filter to each column along the azimuth direction
; plot result
SHADE_SURF, abs(datacompr), XTITLE = "Range", YTITLE = "Azimuth", TITLE='After range compression'
window,1, xs=Nrg, ys=Naz, title='Compressed data along range - dB'
!P.MULTI = 0
tvscl,20*alog10(abs(datacompr))
window, 2, xs=Nrg, ys=Naz, title='Compressed data along range - Phase'
!P.MULTI = 0
loadct,39
tv, bytscl(congrid( atan(datacompr,/ph),Nrg,Naz),-!pi,!pi)
;
;stop
; - compression in azimuth
time_a = time_a-max(time_a)/2				; shift time to symmetric array
mfa = dcomplexarr(Naz)
enva = rect(time_a/tmax)			; design the envelope
mfa = enva*exp(!pi*z*KA*(time_a)^2)		; filter in time domain to compress in azimuth
datacompa = COMPLEXARR(Nrg,Naz)
; plots
window,3, xs=1450, ys=600, title='Processing along azimuth'
!P.MULTI = [0,3,1]
; azimuth - matched filter - phase
plot, time_a/(1e-3), atan(mfa,/ph), yrange=[-5,5],xrange=[-Naz/PRF/2*1e3,Naz/PRF/2*1e3], XTITLE = 'Time [ms]', YTITLE = "Phase [rad]",TITLE='Matched filter, az. - Ph.',xstyle=1,ystyle=1,background=255,color=0
; azimuth - matched filter - frequency response
plot, frec_a/1e3, abs(shift(fft(mfa),Naz/2)),yrange=[0,0.05],xrange=[-PRF/2/1e3,PRF/2/1e3],xstyle=1,ystyle=1,xtitle='Frequency [KHz]',ytitle='[-]',title='Matched filter, az. - Response',background=255,color=0
; data compression
datacompa = COMPLEXARR(Nrg,Naz)
for i=0,Nrg-1 do datacompa(i,*) = CONVOL(data(i,*),transpose(mfa),/edge_zero)  ; apply filter to each column along the range direction
; plot result
SHADE_SURF, abs(datacompa), XTITLE = "Range", YTITLE = "Azimuth", TITLE='After azimuth compression'
!p.multi=0
window,4, xs=Nrg, ys=Naz, title='Compressed data along azimuth - dB'
tvscl,20*alog10(abs(datacompa))

window,5, xs=Nrg, ys=Naz, title='Compressed data along azimuth - Phase'
!P.MULTI = 0
loadct,39
tv, bytscl(congrid( atan(datacompa,/ph),Nrg,Naz),-!pi,!pi)
;
; - compression in range and azimuth
datacompra = COMPLEXARR(Nrg,Naz)
for i=0,Nrg-1 do datacompra(i,*) = CONVOL(datacompr(i,*),TRANSPOSE(mfa),/edge_zero) ;apply filter to each line along the range direction
window,6, xs=Nrg, ys=Naz, title='Compressed data - dB'
tvscl,20*alog10(abs(datacompra))
window,7, xs=500,ys=600
SHADE_SURF, abs(datacompra), XTITLE = "Range", YTITLE = "Azimuth", TITLE='Compressed signal'

; --- HAMMING WINDOWING
; - compression in range
hanr=dblarr(Nrg)
ii=where(env gt 0.5)
han0 = HANNING(n_elements(ii), alpha=0.54)
hanr(ii)=han0
mfrhan = dcomplexarr(Nrg)  
mfrhan = hanr*mfr				; apply hamming window to the range matched filter defined above
datacompr = COMPLEXARR(Nrg,Naz)
for i=0,Naz-1 do datacompr(*,i) = CONVOL(data(*,i),mfrhan,/edge_zero) ;apply filter including hamming window
; - compression in azimuth
hana=dblarr(Naz)
ii=where(enva gt 0.5)
han0 = HANNING(n_elements(ii), alpha=0.54)
hana(ii)=han0
mfahan = hana*mfa				;apply hamming window to the azimuth matched filter defined above
datacompa = COMPLEXARR(Nrg,Naz)
for i=0,Nrg-1 do datacompra(i,*) = CONVOL(datacompr(i,*),TRANSPOSE(mfahan),/edge_zero) ;apply filter including hamming window
; - plots
window,8, xs=Nrg, ys=Naz, title='Compressed data with hanning - dB'
tvscl,20*alog10(abs(datacompra))
window,9, xs=500,ys=600
SHADE_SURF, abs(datacompra), XTITLE = "Range", YTITLE = "Azimuth", TITLE='Compressed signal'


stop

; write_png, '2Dhan.png', TVRD(TRUE=1)


;FILTER IN FREQUENCY DOMAIN
window, 3, xsize=1500, ysize=1500
!P.MULTI = [0,1,2]
;compression in range
mffr = DCOMPLEXARR(512)
mfr = shift(mfr,256)
mffr = FFT(mfr,-1)				;FFT of range matched filter
plot, frec, atan(shift(mffr,256),/ph), XTITLE = "Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the range matched filter in frequency domain'
plot, frec, abs(shift(mffr,256)), XTITLE = "Frequency [Hz]", YTITLE = "Magnitude", TITLE='Magnitude of the range matched filter in frequency domain'

datasp = FFT(data, DIMENSION=1)			;FFT of the data
; plot, frec, atan(datasp(*,100),/ph), XTITLE = "Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the data in frequency domain'
; ; write_png, '2Df1.png', TVRD(TRUE=1)

dataspcompr = DCOMPLEXARR(512,1081)
for i=0,1080 do dataspcompr(*,i) = datasp(*,i)*mffr	;apply filter by multiplying with every single column

datacompr = FFT(dataspcompr, 1, DIMENSION=1)		;inverse FFT

window, 4, xsize=1000, ysize=1500
!P.MULTI = [0,1,2]
SHADE_SURF, abs(datacompr), XTITLE = "Range", YTITLE = "Azimuth", TITLE='Compressed signal in range'

plot, time,20*ALOG10(abs(datacompr(*,540))),xrange=[-2e-7, 2e-7],yrange=[-30,0], XTITLE = "Time [s]", YTITLE = "Magnitude [dB]",TITLE='Magnitude of one compressed range signal in dB'

; write_png, '2Df2.png', TVRD(TRUE=1)


window, 5, xsize=1500, ysize=1500
!P.MULTI = [0,1,2]
;compression in azimuth
mffa = DCOMPLEXARR(1081)
mfa = shift(mfa, 541)
mffa = FFT(mfa, -1)				;FFT of azimuth matched filter
; mffa = shift(mffa,-541)

plot, frec_a, atan(shift(mffa,541),/ph), XTITLE="Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the azimuth matched filter in frequency domain'
plot, frec_a, abs(shift(mffa,541)), XTITLE = "Frequency [Hz]", YTITLE = "Magnitude", TITLE='Magnitude of the azimuth matched filter in frequency domain'

dataspa = FFT(datacompr, DIMENSION=2)		;FFT of range compressed data

; plot, frec_a, atan(dataspa(100,*),/ph), XTITLE = "Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the data in frequency domain'
; write_png, '2Df3.png', TVRD(TRUE=1)
window, 6, xsize=1000, ysize=1500
!P.MULTI = [0,1,2]
dataspcompa = DCOMPLEXARR(512,1081)
for i=0,511 do dataspcompa(i,*) = dataspa(i,0:1080)*mffa	;apply filter by multiplying with every single line

datacompa = FFT(dataspcompa, 1, DIMENSION=2)			;inverse FFT

SHADE_SURF, abs(datacompa), XTITLE = "Range", YTITLE = "Azimuth", TITLE='Compressed signal'

plot, time_a,20*ALOG10(abs(datacompa(256,*))),xrange=[-0.2, 0.2], XTITLE = "Time [s]", YTITLE = "Magnitude [dB]",TITLE='Magnitude of one compressed azimuth signal in dB'

; write_png, '2Df4.png', TVRD(TRUE=1)


;FILTER IN FREQUENCY DOMAIN WITH HAMMING
window, 7, xsize=1500, ysize=1500
!P.MULTI = [0,1,2]
;compression in range
mffrhan = DCOMPLEXARR(512)
mfrhan = shift(mfrhan,256)
mffrh = FFT(mfrhan,-1)				;FFT of range matched filter
plot, frec, atan(shift(mffrh,256),/ph), XTITLE = "Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the range matched filter in frequency domain'
plot, frec, abs(shift(mffrh,256)), XTITLE = "Frequency [Hz]", YTITLE = "Magnitude", TITLE='Magnitude of the range matched filter in frequency domain'

datasp = FFT(data, DIMENSION=1)			;FFT of the data

; plot, frec, atan(datasp(*,100),/ph), XTITLE = "Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the data in frequency domain'
; write_png, '2Dfh1.png', TVRD(TRUE=1)

dataspcompr = DCOMPLEXARR(512,1081)
for i=0,1080 do dataspcompr(*,i) = datasp(*,i)*mffrh	;apply filter by multiplying with every single column

datacompr = FFT(dataspcompr, 1, DIMENSION=1)		;inverse FFT

window, 8, xsize=1000, ysize=1500
!P.MULTI = [0,1,2]
SHADE_SURF, abs(datacompr), XTITLE = "Range", YTITLE = "Azimuth", TITLE='Compressed signal in range'

plot, time,20*ALOG10(abs(datacompr(*,540))),xrange=[-.5e-6, .5e-6],yrange=[-30,0], XTITLE = "Time [s]", YTITLE = "Magnitude [dB]",TITLE='Magnitude of one compressed range signal in dB'

; write_png, '2Dfh2.png', TVRD(TRUE=1)


window, 9, xsize=1500, ysize=1500
!P.MULTI = [0,1,2]
;compression in azimuth
mffahan = DCOMPLEXARR(1081)
mfahan = shift(mfahan, 541)
mffah = FFT(mfahan, -1)				;FFT of azimuth matched filter

plot, frec_a, atan(shift(mffah,541),/ph), XTITLE="Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the azimuth matched filter in frequency domain'
plot, frec_a, abs(shift(mffah,541)), XTITLE = "Frequency [Hz]", YTITLE = "Magnitude", TITLE='Magnitude of the azimuth matched filter in frequency domain'

dataspa = FFT(datacompr, DIMENSION=2)		;FFT of range compressed data

; plot, frec_a, atan(dataspa(100,*),/ph), XTITLE = "Frequency [Hz]", YTITLE = "Phase [rad]",TITLE='Phase of the data in frequency domain'
; write_png, '2Dfh3.png', TVRD(TRUE=1)
window, 10, xsize=1000, ysize=1500
!P.MULTI = [0,1,2]
dataspcompa = DCOMPLEXARR(512,1081)
for i=0,511 do dataspcompa(i,*) = dataspa(i,0:1080)*mffah	;apply filter by multiplying with every single line

datacompa = FFT(dataspcompa, 1, DIMENSION=2)			;inverse FFT

SHADE_SURF, abs(datacompa), XTITLE = "Range", YTITLE = "Azimuth", TITLE='Compressed signal with Hamming'

plot, time_a,20*ALOG10(abs(datacompa(256,*))),xrange=[-0.2, 0.2], XTITLE = "Time [s]", YTITLE = "Magnitude [dB]",TITLE='Magnitude of one compressed azimuth signal in dB with Hamming'

; write_png, '2Dfh4.png', TVRD(TRUE=1)
stop

END