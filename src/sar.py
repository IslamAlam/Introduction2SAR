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
    sar_img_rescale = np.interp(np.abs(sar_img), (np.amin(np.abs(sar_img)), 3*np.mean(np.abs(sar_img))), (0, 255))
    fig = plt.figure(constrained_layout=False, figsize=(10, 10))
    gs1 = fig.add_gridspec(nrows=1, ncols=2)

    f_ax1 = fig.add_subplot(gs1[:, 0])
    if log:
        log_cmp = LogNorm(vmin=100, vmax=np.max(np.abs(sar_img)))
        ax_abs = f_ax1.imshow(np.abs(sar_img), cmap='gray', norm=log_cmp, **kwargs)
    else:
        ax_abs = f_ax1.imshow(np.abs(sar_img_rescale), cmap='gray', **kwargs)
            
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

def dB(y):
    "Calculate the log ratio of y / max(y) in decibel."

    y = np.abs(y)
    y /= y.max()

    return 20 * np.log10(y)


def log_plot_normalized(x, y, ylabel, ax):
    ax.plot(x, dB(y))
    ax.set_ylabel(ylabel)
    ax.grid()


def rect(t, Tp):
    return np.where(abs(t)<=Tp/0.5, 1, 0)


def convolve_axis(in1, in2, axis, **kwargs):
    return np.apply_along_axis(lambda m: np.convolve(m, in2, **kwargs), axis=axis, arr=in1) #  apply filter to each column along the azimuth direction


def convolve_2d(in1, in2, in3, axis, **kwargs):
    convolve_1ax = convolve_axis(in1,          in2=in2, axis=axis[0], **kwargs) #  apply filter to each column along the azimuth direction
    convolve_2ax = convolve_axis(convolve_1ax, in2=in3, axis=axis[1], **kwargs) #  apply filter to each column along the azimuth direction

    return convolve_2ax

def time_freq_plot(tt, frequ, signal, title,  **kwargs):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    gs = GridSpec(4, 3, figure=fig)

    # plot matched filter - real
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tt/(1e-6), np.real(signal))
    ax.set_title(title + " - Real.")
    ax.set_xlabel('Time [ $\mu$s]')
    ax.set_ylabel("Real")

    # plot matched filter - imag
    ax = fig.add_subplot(gs[1, :])
    ax.plot(tt/(1e-6), np.imag(signal))
    ax.set_title(title + " - Imag.")
    ax.set_xlabel('Time [ $\mu$s]')
    ax.set_ylabel("Real")

    # plot frequency response
    ax = fig.add_subplot(gs[2, :])
    ax.plot(tt/(1e-6), np.angle(signal, deg=False))
    ax.set_title(title + " - Ph.")
    ax.set_xlabel('Time [ $\mu$s]')
    ax.set_ylabel("Phase [rad]")

    # section along the range
    ax = fig.add_subplot(gs[3, :])
    ax.plot(frequ/(1e-6), np.absolute(fftshift(fft(signal))))
    ax.set_title(title + ' - Response')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel("[-]")
    plt.show()
    
    return

def cross_hair(x, y, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    horiz = ax.axhline(y, **kwargs)
    vert = ax.axvline(x, **kwargs)
    return horiz, vert

def add_marker(x, y, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    horiz = ax.scatter(x*2, y,  marker='<', **kwargs)
    vert = ax.scatter(x, -5,  marker='v', **kwargs)
    return horiz, vert


def add_marker_text(x, y, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    horiz = ax.text(x*2, y,  s='< Range section', horizontalalignment='left', **kwargs)
    vert = ax.text(x, -5,  s='Azimuth section \nv',  horizontalalignment='center', **kwargs)
    return horiz, vert


def surface_plot(radar_image, title, xlabel, ylabel, zlabel, **kwargs):
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    fig = plt.figure(constrained_layout=True, figsize=(12,12))
    gs = GridSpec(3, 3, figure=fig)


    # figure, ax = plt.subplots(figsize=(12, 12))
    # ax.axis("off")
    # figure.patch.set_visible(False)
    ax = fig.add_subplot(gs[0:2, :], projection='3d')

    # ax = figure.add_subplot(2, 2, 1, projection='3d')
    Z2 = np.absolute((radar_image)) # dB
    Z2 = Z2 / (np.max(Z2)- np.min(Z2))

    X2, Y2 = np.meshgrid(range(Z2.shape[1]), range(Z2.shape[0]))

    X1 = np.reshape(X2, -1)
    Y1 = np.reshape(Y2, -1)
    #ax = plt.axes(projection='3d')
    ax.plot_surface(X2, Y2, (Z2), cmap='jet')

    from matplotlib import cm
    # Normalize the colors based on Z value
    norm = plt.Normalize(Z2.min(), Z2.max())
    colors = cm.jet(norm(Z2))
    # ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X2, Y2, Z2, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))

    cset = ax.contourf(X2, Y2, Z2, zdir='z', offset=(np.max(Z2)-np.min(Z2))*-1, cmap=cm.viridis)
    cset = ax.contourf(X2, Y2, Z2, zdir='x', offset=-5, cmap=cm.viridis)
    cset = ax.contourf(X2, Y2, Z2, zdir='y', offset=Z2.shape[1]+5, cmap=cm.viridis)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_xlabel(xlabel)
    ax.set_xlim((-5, Z2.shape[1]))

    ax.set_ylabel(ylabel)
    ax.set_ylim((-5, Z2.shape[0]))

    ax.set_zlabel(zlabel)
    ax.set_zlim((np.max(Z2)-np.min(Z2))*-1, np.max(Z2))


    #ax.contourf(dB(radar_slice), contours, cmap='magma_r')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Amp (Normalized)")

    ### show sar image
    ax = fig.add_subplot(gs[2, 0])
    log_cmp = LogNorm(vmin=100, vmax=np.max(np.abs(radar_image)))

    sar_img = ax.imshow(np.absolute((radar_image)), cmap='jet', norm=log_cmp, **kwargs)
    # fig.colorbar(sar_img, ax=ax, shrink=0.9)
    # cross_hair(Z2.shape[1]/2, Z2.shape[0]/2, color='red', xmin=-0.1, xmax=.9)
    add_marker_text(Z2.shape[1]/2, Z2.shape[0]/2, color='red')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # zoom out
    #I want to select the x-range for the zoomed region. I have figured it out suitable values
    x1, x2 = Z2.shape[1]*.48, Z2.shape[1]*.5

    # select y-range for zoomed region
    y1, y2 = Z2.shape[0]*.48, Z2.shape[0]*.52

    # Make the zoom-in plot:
    axins2 = zoomed_inset_axes(ax, 6, loc=1)  # zoom = 6
    axins2.imshow(Z2, cmap="gray", **kwargs)
    for axis in ['top','bottom','left','right']:
        axins2.spines[axis].set_linewidth(3)
        axins2.spines[axis].set_color('r')
    # sub region of the original image
    # x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    # fix the number of ticks on the inset axes
    axins2.yaxis.get_major_locator().set_params(nbins=7)
    axins2.xaxis.get_major_locator().set_params(nbins=7)

    plt.setp(axins2.get_xticklabels(), visible=False)
    plt.setp(axins2.get_yticklabels(), visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="r")

    # section along the range
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(Z2[int(Z2.shape[0]/2),:])
    ax.set_xlabel(xlabel+" section")
    ax.set_ylabel(zlabel)

    # section along the azimuth
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(Z2[:,int(Z2.shape[1]/2)])
    ax.set_xlabel(ylabel+" section")
    ax.set_ylabel(zlabel)

    return fig

def pad_zeros_axis(new_n_sample, vector):

    pad_len = round((new_n_sample-vector.shape[0])/2)
    pad_len_right = new_n_sample-(pad_len+vector.shape[0])
    print("Pad to left side", pad_len, "pad to right side", new_n_sample-(pad_len+vector.shape[0]))
    vector = np.pad(vector, (pad_len, pad_len_right), 'constant', constant_values=(0, 0))
    # vector = np.roll(vector, round(new_n_sample/2))
    return vector