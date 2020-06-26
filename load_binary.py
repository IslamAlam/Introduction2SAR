# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:59:04 2014

@author: alon_al
"""
from __future__ import print_function
import struct
import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from subprocess import call
from os import listdir
from os.path import isfile, join


# Read all the floats in a binary file
# Returns a python array
def floats_from_file(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    return struct.unpack("f" * (len(fileContent) // 4), fileContent)

# Read all the doubles in a binary file
# Returns a python array
def doubles_from_file(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    return struct.unpack("d" * (len(fileContent) // 8), fileContent)

# Read all the ints in a binary file
# Returns a python array
def ints_from_file(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    return struct.unpack("i" * (len(fileContent) // 4), fileContent)

# Read all the longs in a binary file
# Returns a python array
def longs_from_file(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    return struct.unpack("q" * (len(fileContent) // 8), fileContent)


############################ numpy functions ##################################

# Load a float image from a binary file
# Returns a numpy 2D array
def load_float_image(filename, rows, cols):
    img=numpy.array(floats_from_file(filename))
    img=img.reshape(rows, cols)
    return img

# Load a double image from a binary file
# Returns a numpy 2D array
def load_double_image(filename, rows, cols):
    img=numpy.array(doubles_from_file(filename))
    img=img.reshape(rows, cols)
    return img

# Load a complex float image from a binary file
# Returns a numpy 2D array
def load_complex_float_image(filename, rows, cols):
    img=numpy.array(floats_from_file(filename))
    img=img.reshape(rows, cols*2)
    return img[:,0:cols*2:2] + 1j*img[:,1:cols*2:2]
    
# Load a complex double image from a binary file
# Returns a numpy 2D array
def load_complex_double_image(filename, rows, cols):
    img=numpy.array(doubles_from_file(filename))
    img=img.reshape(rows, cols*2)
    return img[:,0:cols*2:2] + 1j*img[:,1:cols*2:2]
    
# Load a complex double matrix image from a binary file
# That is, a rows x cols image containing a m_rows x m_cols matrix on each position.
# Returns a numpy 2D array
def load_complex_double_matrix_image(filename, rows, cols, m_rows, m_cols):
    img=numpy.array(doubles_from_file(filename))
    img=img.reshape(rows, cols*2*m_rows*m_cols)
    img=img[:,0:cols*2*m_rows*m_cols:2] + 1j*img[:,1:cols*2*m_rows*m_cols:2]
    return img.reshape(rows, cols, m_rows, m_cols)

# Load an int image from a binary file
# Returns a numpy 2D array
def load_int_image(filename, rows, cols):
    img=numpy.array(ints_from_file(filename))
    img=img.reshape(rows, cols)
    return img

# Load a long image from a binary file
# Returns a numpy 2D array
def load_long_image(filename, rows, cols):
    img=numpy.array(longs_from_file(filename))
    img=img.reshape(rows, cols)
    return img
    
# Read all the floats in a binary file with size (DAT file)
# DAT files contain two C integers as header, containing rows and cols inforamation
# Returns a numpy 2D array
def floats_from_file_with_size(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    rows = struct.unpack("i", fileContent[:4])
    cols = struct.unpack("i", fileContent[4:8])
    print(rows,cols)
    return numpy.array(struct.unpack("f" * (rows*cols), fileContent[8:]),dtype=numpy.float32).reshape(cols,rows)

# Read all the floats in a binary file with size (DAT file) swaping endian
# DAT files contain two C integers as header, containing rows and cols inforamation
# Returns a numpy 2D array
def floats_from_file_with_size_swap_endian(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    rows = struct.unpack("i", fileContent[:4])
    cols = struct.unpack("i", fileContent[4:8])
    rows = numpy.array(rows, dtype=numpy.int32).byteswap(True)[0]
    cols = numpy.array(cols, dtype=numpy.int32).byteswap(True)[0]
    print(rows,cols)
    return numpy.array(struct.unpack("f" * (rows*cols), fileContent[8:8+rows*cols*4]),dtype=numpy.float32).byteswap(True).reshape(cols,rows)

# Read all the float complex in a binary file with size (DAT file) swaping endian
# DAT files contain two C integers as header, containing rows and cols inforamation
# Returns a numpy 2D array
def complexs_from_file_with_size_swap_endian(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    rows = struct.unpack("i", fileContent[:4])
    cols = struct.unpack("i", fileContent[4:8])
    rows = numpy.array(rows, dtype=numpy.int32).byteswap(True)[0]
    cols = numpy.array(cols, dtype=numpy.int32).byteswap(True)[0]
    print(rows,cols)
    data = numpy.array(struct.unpack("f" * (rows*cols*2), fileContent[8:]),dtype=numpy.float32).byteswap(True)
    data = (data[0::2] + 1j*data[1::2]).astype(numpy.complex64)
    return data.reshape(cols,rows)

def doubles_from_file_with_size_1D_swap_endian(filename):
    """
    Read all the doubles in a binary 1D file with size (DAT file) swaping
    endian.
    These DAT files contain one C integer as header, containing the number of
    smaples inforamation
    
    Returns a numpy 1D array of float64 type
    """
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    rows = struct.unpack("i", fileContent[:4])
    rows = numpy.squeeze(numpy.asarray(rows, dtype=numpy.int32).byteswap(True))
    return numpy.asarray(struct.unpack("d" * (rows), fileContent[4:]), dtype=numpy.float64).byteswap(True)


# Save a numpy array into a dat format file for IDL (swaping endian)    
def save_dat_float_array_swap_endian(nparr, filename):
    shp = numpy.array(nparr.shape[::-1], dtype=numpy.int32).byteswap(True);
    f = open(filename, mode="wb")  # b is important -> binary
    f.write(shp.tostring())
    f.write(nparr.byteswap().tostring())
    f.close()

# Load a Pauli image from a full-pol T3 folder in PolSARPro format
# NOTE: The config.txt file is not read. Size must be provided.
# Returns a numpy 3D array representing the Pauli reflectivity, not an RGB image
# NOTE: The returned image is not RGB displayable, as it is not normalized
def load_PauliRGB_from_T3(folderT3, rows, cols, acquisition=0):
    r=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+2) + str(acquisition*3+2) + ".bin"))
    g=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+3) + str(acquisition*3+3) + ".bin"))
    b=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+1) + str(acquisition*3+1) + ".bin"))
    rgb=numpy.empty((rows,cols,3))
    rgb[:,:,0] = r.reshape(rows,cols)
    rgb[:,:,1] = g.reshape(rows,cols)
    rgb[:,:,2] = b.reshape(rows,cols)
    return numpy.sqrt(rgb)
    

def get_PauliRGB_from_T3(T3):
    """
    Return a Pauli RGB (NOT NORMALIZED) from a T3 matrix data (rows, cols, 3, 3)
    """
    rows = T3.shape[0]
    cols = T3.shape[1]
    r=numpy.sqrt(numpy.abs(numpy.array(T3[...,1,1])))
    g=numpy.sqrt(numpy.abs(numpy.array(T3[...,2,2])))
    b=numpy.sqrt(numpy.abs(numpy.array(T3[...,0,0])))
    rgb=numpy.zeros((rows,cols,3))
    rgb[:,:,0] = r.reshape(rows,cols)
    rgb[:,:,1] = g.reshape(rows,cols)
    rgb[:,:,2] = b.reshape(rows,cols)
    return rgb

# Load a Pauli image from a full-pol T3 folder in PolSARPro format
# NOTE: The config.txt file is not read. Size must be provided.
# Returns a numpy 3D array (normalized RGB image)
# The returned image is ready for displaying or saving as an image
def load_PauliRGB_adjusted_from_T3(folderT3, rows, cols, acquisition=0):
    r=numpy.sqrt(numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+2) + str(acquisition*3+2) + ".bin")))
    g=numpy.sqrt(numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+3) + str(acquisition*3+3) + ".bin")))
    b=numpy.sqrt(numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+1) + str(acquisition*3+1) + ".bin")))
    rgb=numpy.zeros((rows,cols,3))
    rgb[:,:,0] = r.reshape(rows,cols)
    rgb[:,:,1] = g.reshape(rows,cols)
    rgb[:,:,2] = b.reshape(rows,cols)
    rgb = adjust_rgb_per_channel(rgb)   # Normalize RGB
    return rgb

# Adjust the Pauli reflectivity 3D image to conform a displayable RGB image
# The dynamic range of each channel isadjusted independently, from 0 to a 
#   given factor of the mean (default 3)
# NOTE: This normalization produce the most eye-attractive results but it
#   does not reflect the real backscattered power.
def adjust_rgb_per_channel(rgb, factor=3):
    r=rgb[:,:,0]
    g=rgb[:,:,1]
    b=rgb[:,:,2]
    rm = numpy.nanmean(r)
    gm = numpy.nanmean(g)
    bm = numpy.nanmean(b)
    r/=factor*rm
    g/=factor*gm
    b/=factor*bm
    rgb[rgb>1]=1
    return rgb

# Adjust the Pauli reflectivity 3D image to conform a displayable RGB image
# The dynamic range of each channel isadjusted together, from 0 to a 
#   given factor of the mean (default 2)
# NOTE: This normalization produce less eye-attractive results but it
#   a more realistic representation of the real backscattered power.
def adjust_rgb_together(rgb, factor=2):
    m=numpy.nanmean(rgb)
    rgb/=factor*m
    rgb[rgb>1]=1
    return rgb

# Adjust the Pauli reflectivity 3D image to conform a displayable RGB image
# The power information of each pixel is removed by dividing the Pauli vector
#   by the trace of the T matrix (sum of the vector).
# NOTE: This normalization just gives information of the type of scattering
#   in the Pauli basis, but about reflectivity itself.
def normalize_rgb_per_pixel_trace(rgb):
    m=numpy.sum(rgb,2)
    rgb[:,:,0]/=m
    rgb[:,:,1]/=m
    rgb[:,:,2]/=m
    return rgb

# Adjust the Pauli reflectivity 3D image to conform a displayable RGB image
# The power information of each pixel is removed by dividing the Pauli vector
#   by the norm of the vector.
# NOTE: This normalization just gives information of the type of scattering
#   in the Pauli basis, but about reflectivity itself. Produces more vivid
#   colors than the trace normalization.
def normalize_rgb_per_pixel_norm(rgb):
    n=numpy.linalg.norm(rgb,2,2)
    rgb[:,:,0]/=n
    rgb[:,:,1]/=n
    rgb[:,:,2]/=n
    return rgb

# Produces 3 separated images for each RGB channel intensity of the given image.
# Each of the image has the _red, _green and _blue suffix and is represented
#   in the corresponding color.
def decompose_RGB_channels(imagefile):
    ext = ((imagefile[::-1])[0:imagefile[::-1].find('.')+1])[::-1]
    name = ((imagefile[::-1])[imagefile[::-1].find('.')+1:])[::-1]
    rgb = mpimg.imread(imagefile)
    assert len(rgb.shape)==3    # Ensure it is an RGB or RGBA image, not grayscale
    rgbt = numpy.zeros(rgb.shape)
    if(rgb.shape[2] == 4):      # If the image has alfa channel then copy it
        rgbt[...,3] = rgb[...,3]
    rgbt[...,0] = rgb[...,0]
    plt.imsave(arr=rgbt,fname=name+'_red' + ext)
    rgbt[...,0] = 0
    rgbt[...,1] = rgb[...,1]
    plt.imsave(arr=rgbt,fname=name+'_green' + ext)
    rgbt[...,1] = 0
    rgbt[...,2] = rgb[...,2]
    plt.imsave(arr=rgbt,fname=name+'_blue' + ext)
    rgbt=None

# Load a file from the Thomas x-bragg simulator T3 output.
# The data format is dat from IDL (swaping byte ordering), with 3 dimensions.
# The headers contains 5 integers (1 x 1 x samples x rows x cols). The first
# two elements (1 x 1) are not readed and assumed as 1.
# Returns a numpy 3D array (cube of complex values)
def load_T3_thomas_file(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    samples = struct.unpack("i", fileContent[8:12])
    rows = struct.unpack("i", fileContent[12:16])
    cols = struct.unpack("i", fileContent[16:20])
    rows = numpy.array(rows, dtype=numpy.int32).byteswap(True);
    cols = numpy.array(cols, dtype=numpy.int32).byteswap(True);
    samples = numpy.array(samples, dtype=numpy.int32).byteswap(True);
    print(rows,cols,samples)
    img=numpy.array(struct.unpack("d" * (rows*cols*samples*2), fileContent[20:]),dtype=numpy.float64).byteswap(True)
    img=img.reshape(cols,rows, samples*2)
    return img[:,:,0:samples*2:2] + 1j*img[:,:,1:samples*2:2]
    
# Load a complete T3 image from a Thomas x-bragg simulator output.
# This implies reading a t11, t12, t22 and t33 files in DAT format (with size
# headers) from IDL (swaping byte ordering). No missing files!!!
# Returns a numpy 5D array (cube of T3 matrices)
def load_T3_thomas_sim(folderT3, sufix=""):
    t11 = load_T3_thomas_file(folderT3 + str("/t11") + sufix)
    result = numpy.empty((t11.shape[0], t11.shape[1], t11.shape[2], 3, 3),dtype=complex)
    result[:,:,:,0,0]=t11
    t11 = None
    result[:,:,:,1,1]= load_T3_thomas_file(folderT3 + str("/t22") + sufix)
    result[:,:,:,2,2]= load_T3_thomas_file(folderT3 + str("/t33") + sufix)
    result[:,:,:,0,1]= load_T3_thomas_file(folderT3 + str("/t12") + sufix)
    result[:,:,:,0,2]= load_T3_thomas_file(folderT3 + str("/t13") + sufix)
    result[:,:,:,1,2]= load_T3_thomas_file(folderT3 + str("/t23") + sufix)
    result[:,:,:,1,0]= numpy.conj(result[:,:,:,0,1])
    result[:,:,:,2,0]= numpy.conj(result[:,:,:,0,2])
    result[:,:,:,2,1]= numpy.conj(result[:,:,:,1,2])
    return result
    
# Load a complete T3 image from a Thomas x-bragg simulator output.
# This implies reading a t11, t12, t22 and t33 files in DAT format (with size
# headers) from IDL (swaping byte ordering). The missing t13 and t23 files are
# considered as 0.
# Returns a numpy 5D array (cube of T3 matrices)
def load_T3_thomas_sim_zeros(folderT3, sufix=""):
    t11 = load_T3_thomas_file(folderT3 + str("/t11") + sufix)
    result = numpy.empty((t11.shape[0], t11.shape[1], t11.shape[2], 3, 3),dtype=complex)
    result[:,:,:,0,0]=t11
    t11 = None
    result[:,:,:,1,1]= load_T3_thomas_file(folderT3 + str("/t22") + sufix)
    result[:,:,:,2,2]= load_T3_thomas_file(folderT3 + str("/t33") + sufix)
    result[:,:,:,0,1]= load_T3_thomas_file(folderT3 + str("/t12") + sufix)
    result[:,:,:,0,2]= numpy.zeros(result.shape[0:3],dtype=complex) # Missing files as 0
    result[:,:,:,1,2]= numpy.zeros(result.shape[0:3],dtype=complex) # Missing files as 0
    result[:,:,:,1,0]= numpy.conj(result[:,:,:,0,1])
    result[:,:,:,2,0]= numpy.conj(result[:,:,:,0,2])
    result[:,:,:,2,1]= numpy.conj(result[:,:,:,1,2])
    return result

# Load a complete T3 image from a full-pol T3 folder in PolSARPro format
# NOTE: The config.txt file is not read. Size must be provided.
# Returns a numpy 4D array (image of T3 matrices)
def load_T3_image(folderT3, rows, cols):
    result = numpy.empty((rows, cols, 3, 3),dtype=complex)
    result[:,:,0,0]=numpy.array(floats_from_file(folderT3 + "/T11.bin")).reshape(rows,cols)
    result[:,:,1,1]=numpy.array(floats_from_file(folderT3 + "/T22.bin")).reshape(rows,cols)
    result[:,:,2,2]=numpy.array(floats_from_file(folderT3 + "/T33.bin")).reshape(rows,cols)
    result[:,:,0,1]=numpy.array(floats_from_file(folderT3 + "/T12_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + "/T12_imag.bin")).reshape(rows,cols)
    result[:,:,0,2]=numpy.array(floats_from_file(folderT3 + "/T13_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + "/T13_imag.bin")).reshape(rows,cols)
    result[:,:,1,2]=numpy.array(floats_from_file(folderT3 + "/T23_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + "/T23_imag.bin")).reshape(rows,cols)
    result[:,:,1,0]=numpy.conj(result[:,:,0,1]).reshape(rows,cols)
    result[:,:,2,0]=numpy.conj(result[:,:,0,2]).reshape(rows,cols)
    result[:,:,2,1]=numpy.conj(result[:,:,1,2]).reshape(rows,cols)
    return result

# Load a complete T3 image from a given acquisition number of a full-pol T3
# folder in PolSARPro format (usually from BPT time series output)
# NOTE: The config.txt file is not read. Size must be provided.
# Returns a numpy 4D array (image of T3 matrices)    
def load_T3_image_acquisition(folderT3, rows, cols, acquisition):
    result = numpy.empty((rows, cols, 3, 3),dtype=complex)
    result[:,:,0,0]=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+1) + str(acquisition*3+1) + ".bin")).reshape(rows,cols)
    result[:,:,1,1]=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+2) + str(acquisition*3+2) + ".bin")).reshape(rows,cols)
    result[:,:,2,2]=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+3) + str(acquisition*3+3) + ".bin")).reshape(rows,cols)
    result[:,:,0,1]=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+1) + str(acquisition*3+2) + "_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+1) + str(acquisition*3+2) + "_imag.bin")).reshape(rows,cols)
    result[:,:,0,2]=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+1) + str(acquisition*3+3) + "_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+1) + str(acquisition*3+3) + "_imag.bin")).reshape(rows,cols)
    result[:,:,1,2]=numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+2) + str(acquisition*3+3) + "_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + "/T" + str(acquisition*3+2) + str(acquisition*3+3) + "_imag.bin")).reshape(rows,cols)
    result[:,:,1,0]=numpy.conj(result[:,:,0,1]).reshape(rows,cols)
    result[:,:,2,0]=numpy.conj(result[:,:,0,2]).reshape(rows,cols)
    result[:,:,2,1]=numpy.conj(result[:,:,1,2]).reshape(rows,cols)
    return result
    
# Load a complete matrix image from a given acquisition number of a full-pol T3
# folder in PolSARPro format (usually from BPT time series output)
# This functions allows to specify the matrix_prefix. This allows to load T3,
# C3 or Y3 matrices, for instance.
# NOTE: The config.txt file is not read. Size must be provided.
# Returns a numpy 4D array (image of matrices)    
def load_PSP_image_acquisition(folderT3, rows, cols, acquisition, matrix_prefix="T"):
    result = numpy.empty((rows, cols, 3, 3),dtype=complex)
    result[:,:,0,0]=numpy.array(floats_from_file(folderT3 + str("/") + matrix_prefix + str(acquisition*3+1) + str(acquisition*3+1) + ".bin")).reshape(rows,cols)
    result[:,:,1,1]=numpy.array(floats_from_file(folderT3 + str("/") + matrix_prefix + str(acquisition*3+2) + str(acquisition*3+2) + ".bin")).reshape(rows,cols)
    result[:,:,2,2]=numpy.array(floats_from_file(folderT3 + str("/") + matrix_prefix + str(acquisition*3+3) + str(acquisition*3+3) + ".bin")).reshape(rows,cols)
    result[:,:,0,1]=numpy.array(floats_from_file(folderT3 + str("/") + matrix_prefix + str(acquisition*3+1) + str(acquisition*3+2) + "_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + matrix_prefix + str(acquisition*3+1) + str(acquisition*3+2) + "_imag.bin")).reshape(rows,cols)
    result[:,:,0,2]=numpy.array(floats_from_file(folderT3 + str("/") + matrix_prefix + str(acquisition*3+1) + str(acquisition*3+3) + "_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + matrix_prefix + str(acquisition*3+1) + str(acquisition*3+3) + "_imag.bin")).reshape(rows,cols)
    result[:,:,1,2]=numpy.array(floats_from_file(folderT3 + str("/") + matrix_prefix + str(acquisition*3+2) + str(acquisition*3+3) + "_real.bin")).reshape(rows,cols) + 1j * numpy.array(floats_from_file(folderT3 + matrix_prefix + str(acquisition*3+2) + str(acquisition*3+3) + "_imag.bin")).reshape(rows,cols)
    result[:,:,1,0]=numpy.conj(result[:,:,0,1]).reshape(rows,cols)
    result[:,:,2,0]=numpy.conj(result[:,:,0,2]).reshape(rows,cols)
    result[:,:,2,1]=numpy.conj(result[:,:,1,2]).reshape(rows,cols)
    return result

# Generates image links (fs soft links) for the different acquisitions of a
# time series dataset (usually from BPT time series output) in order to
# generate PolSARPro compatible folders for each acquisition.
# This function calls the OS 'ln' command (usually does not work under Windows)
def generate_image_links(basefolder, subfolder='T3'):
    datafolder = join(basefolder, subfolder)
    imgfolder = join(basefolder, 'images')
    bin_files=[f for f in listdir(datafolder) if isfile(join(datafolder,f)) and f.endswith('.bin')]
    call(['mkdir',imgfolder])
    img_number = 0
    while 'T' + str(img_number * 3 +1) + str(img_number * 3 +1) + '.bin' in bin_files:
        img_subfolder = join(imgfolder, str(img_number + 1))
        call(['mkdir', img_subfolder])
        img_subfolder = join(img_subfolder, 'T3')
        call(['mkdir', img_subfolder])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +1) + str(img_number * 3 +1) + '.bin'),
            join(img_subfolder, 'T11.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +2) + str(img_number * 3 +2) + '.bin'),
            join(img_subfolder, 'T22.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +3) + str(img_number * 3 +3) + '.bin'),
            join(img_subfolder, 'T33.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +1) + str(img_number * 3 +2) + '_real.bin'),
            join(img_subfolder, 'T12_real.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +1) + str(img_number * 3 +2) + '_imag.bin'),
            join(img_subfolder, 'T12_imag.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +1) + str(img_number * 3 +3) + '_real.bin'),
            join(img_subfolder, 'T13_real.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +1) + str(img_number * 3 +3) + '_imag.bin'),
            join(img_subfolder, 'T13_imag.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +2) + str(img_number * 3 +3) + '_real.bin'),
            join(img_subfolder, 'T23_real.bin')])
        call(['ln', join(datafolder, 'T' + str(img_number * 3 +2) + str(img_number * 3 +3) + '_imag.bin'),
            join(img_subfolder, 'T23_imag.bin')])
        call(['cp', join(datafolder, 'config.txt'), join(img_subfolder, 'config.txt')])
        img_number += 1
