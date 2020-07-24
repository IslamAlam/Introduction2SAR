import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import scipy



# implement functions to convert SAR data from decibel units to linear units and back again
def decibel_to_linear(band):
     # convert to linear units
    return np.power(10,np.array(band)/10)


def linear_to_decibel(band):
    return 10*np.log10(band)


def colorize(z):
    from colorsys import hls_to_rgb

    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    ll = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, ll, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    return c


def filter_image(im_noise, n_window):
    from scipy import signal
    im_med = signal.wiener(im_noise, (n_window,n_window))
    return im_med


def cross_sar_image(im1, im2):

    if np.all(im1 == im2):
        raise print('Im1 is the same as Im2')

    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    # im1_gray = np.sum(im1.astype('float'), axis=2)
    # im2_gray = np.sum(im2.astype('float'), axis=2)

    # get rid of the averages, otherwise the results are not good
    # im1_gray -= np.mean(im1)
    # im2_gray -= np.mean(im2)

    # im1_corr = scipy.signal.fftconvolve(im1, im2[::-1,::-1], mode='same')
    # im2_corr = scipy.signal.fftconvolve(im2, im1[::-1,::-1], mode='same')
    # im_corr = scipy.signal.fftconvolve(im2, img2, mode='same')

    # y1, x1 = np.unravel_index(np.argmax(im1_corr), im1_corr.shape)  # find the match
    # y2, x2 = np.unravel_index(np.argmax(im2_corr), im2_corr.shape)  # find the match
    # np.unravel_index(np.argmax(corr_img), corr_img.shape)

    # --- determine the pixel shift -----------------------------
    #

    # discrete fast fourier transformation and complex conjugation of image 2
    #
    image1FFT = np.fft.fft2(np.abs(im1))
    image2FFT = np.conjugate( np.fft.fft2(np.abs(im2)) )

    # inverse fourier transformation of product -> equal to cross correlation
    #
    imageCCor = np.real( np.fft.ifft2( (image1FFT*image2FFT) ) )

    # Shift the zero-frequency component to the center of the spectrum
    #
    imageCCorShift = np.fft.fftshift(imageCCor)
    f, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(imageCCorShift)
    # determine the distance of the maximum from the center
    #
    row, col = im1.shape

    yShift, xShift = np.unravel_index(np.argmax(imageCCorShift), (row, col))

    yShift -= int(row/2)
    xShift -= int(col/2)

    print("shift of image1 in x-direction [pixel]: " + str(xShift))
    print("shift of image1 in y-direction [pixel]: " + str(yShift))

    # calculate the correlation image; note the flipping of onw of the images
    return imageCCorShift, xShift, yShift


def plot_sar(sar_img, title, log=True, **kwargs):
    fig = plt.figure(constrained_layout=False, figsize=(10, 10))
    gs1 = fig.add_gridspec(nrows=1, ncols=2)

    f_ax1 = fig.add_subplot(gs1[:, 0])
    if log:
        log_cmp = LogNorm(vmin=10, vmax=np.max(np.abs(sar_img)))
        ax_abs = f_ax1.imshow(np.abs(sar_img), cmap='gray', norm=log_cmp, **kwargs)
    else:
        ax_abs = f_ax1.imshow(np.abs(sar_img), cmap='gray', **kwargs)
            
    f_ax1.set_title("Absolute")
    fig.colorbar(ax_abs, ax=f_ax1, shrink=0.4)

    sar_angle = np.mod(np.angle(sar_img, deg=False), 2*np.pi)
    sar_angle = np.angle(sar_img, deg=False)
    f_ax2 = fig.add_subplot(gs1[:, 1])
    if log:
        log_cmp = LogNorm(vmax=np.max(sar_angle))
        ax_ang = f_ax2.imshow((sar_angle), cmap='rainbow', **kwargs)#, cmap='rainbow', **kwargs)
    else:
        ax_ang = f_ax2.imshow((sar_angle), cmap='rainbow', **kwargs)
    f_ax2.set_title("Phase (angle)")
    # fig.colorbar(ax_ang, ax=f_ax2, shrink=0.6)
    cbar = fig.colorbar(ax_ang, ax=f_ax2, ticks=[-np.pi*.99, 0, np.pi*.99],
                        orientation='vertical', shrink=0.4)
    cbar.ax.set_yticklabels(['$- \pi$', '0', '$\pi$'])  # horizontal colorbar

    fig.suptitle('Plot of ' + title, fontsize=16)

    return


# use np.concatenate and np.full by chrisaycock
def shift_arr(arr, num, fill_value=np.nan):
    if len(num) == 0:
        if num >= 0:
            return np.concatenate((np.full(num, fill_value), arr[:-num]))
        else:
            return np.concatenate((arr[-num:], np.full(-num, fill_value)))
    else:
        if num[0] >= 0:
            shiftx = np.concatenate((np.full((arr.shape[0], num[0]) , fill_value), arr[:,num[0]:]), axis=1)
        else:
            shiftx = np.concatenate((arr[:,-num[0]:], np.full((arr.shape[0], -num[0]) , fill_value)), axis=1)
        if num[1] >= 0:
            return  np.concatenate((shiftx[num[1]:, :], np.full((num[1], arr.shape[1]) , fill_value)), axis=0)
        else:
            return np.concatenate((np.full((-num[1], arr.shape[1]) , fill_value), shiftx[-num[1]:, :]), axis=0)
        
        
        
def coherence_fun(E1, E2, n_window=5, plot=True, **kwargs):
    mean_filter = np.ones((n_window, n_window))
    # E1_filter = scipy.signal.fftconvolve(E1, mean_filter, mode='same')
    # E2_filter = scipy.signal.fftconvolve(E2, mean_filter, mode='same')

    numer = scipy.signal.convolve( E1 * np.conj(E2), mean_filter, mode='same') / np.sum(mean_filter)
    denom = scipy.signal.convolve( np.sqrt(np.abs(np.power(E1,2)) * np.abs(np.power(E2,2))), mean_filter, mode='same') / np.sum(mean_filter)
    
    gamma = numer / denom
    if plot:
        plot_sar(sar_img=(gamma),
         aspect=gamma.shape[1]/gamma.shape[0]*2, **kwargs)
    print("gamma shape:", gamma.shape, "gamma min:", np.abs(gamma).min(), "gamma max:", np.abs(gamma).max())
    return gamma


def plot_spectrum(im_fft):
    f, ax = plt.subplots(figsize=(10, 10))
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    ax_h = ax.imshow(np.abs(im_fft), norm=LogNorm(vmin=5), cmap='viridis',
                     aspect=im_fft.shape[1]/im_fft.shape[0])

    f.colorbar(ax_h, ax=ax, shrink=0.8)
    
    
def shift_flatearth(im_inter_raw):
    interf_img_fft = np.fft.fft2(im_inter_raw)
    interf_img_fftshift = np.fft.fftshift(interf_img_fft)

    row, col = interf_img_fftshift.shape

    yshift, xshift = np.unravel_index(np.argmax(np.abs(interf_img_fftshift)), (row, col))
    print("Fringe Max value = %s @x: %s, @y: %s"% (np.argmax((interf_img_fftshift)), xloc, yloc))
    # shift the image in the frequency domain to remove the flat earth
    im_flat_fft_roll = np.roll(np.roll(interf_img_fftshift, (xshift), axis=1), yshift, axis=0) 
    plot_spectrum(im_flat_fft_roll)

    im_flat_roll_inv = np.fft.ifft2(im_flat_fft_roll)
    plot_sar(sar_img=(im_flat_roll_inv), title="Filtered SAR Image Interferogramm Freq Shift", aspect=im_inter_raw.shape[1]/im_inter_raw.shape[0]*2)
    return im_flat_roll_inv


def cal_interferogram(im1, im2):
    interf_img = im1[1]*np.conj(im2)
    plot_sar(sar_img=(interf_img), title="SAR Image Interferogramm",
             aspect=im1.shape[1]/im1.shape[0]*2)
    interf_img_fft = np.fft.fft2(interf_img)
    interf_img_fftshift = np.fft.fftshift(interf_img_fft)

    row, col = interf_img_fftshift.shape

    yshift, xshift = np.unravel_index(np.argmax(np.abs(interf_img_fftshift)), (row, col))
    print("Fringe Max value = %s @x: %s, @y: %s"% (np.argmax((interf_img_fftshift)), xshift, yshift))
    
    im_flat_roll_inv = shift_flatearth(im_fft_raw, (-xshift, -yshift))
    
    return im_flat_roll_inv