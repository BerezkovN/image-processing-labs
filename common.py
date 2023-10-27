from functools import cache
import cv2

import os
from tkinter import filedialog

from image_viewer import ImageViewer
from gui_elements import MainWindow

import numpy as np
import numba
from numba import jit, prange
numba.config.NUMBA_NUM_THREADS = os.cpu_count()

def open_image(main_window):
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=(
            ("JPEG files", "*.jpeg;*.jpg"),
            ("PNG files", "*.png"),
            ("all files", "*.*"),
        ),
    )
    if not file_path or not os.path.exists(file_path):
        return False
    main_window.image = cv2.imread(file_path)
    main_window.image_viewer.set_image(main_window.image)
    return True


def save_image_viewer(image_viewer):
    if image_viewer.image is not None:
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            filetypes=(
                ("JPEG files", "*.jpeg;*.jpg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff;*.tif"),
                ("all files", "*.*"),
            ),
            defaultextension=".png",
        )
        if file_path:
            cv2.imwrite(file_path, image_viewer.image)
            return True
    return False


def save_image(main_window):
    save_image(main_window.image_viewer)
    

@jit(nopython=True, parallel=True, nogil=True)
def core_cvtColorBGRtoGrayBGR(image):
    h, w, _ = image.shape
    output = np.empty((h, w, 3), dtype=np.uint8)
    
    for i in prange(h):
        for j in prange(w):
            gray_val = (image[i, j, 0] + image[i, j, 1] + image[i, j, 2]) // 3
            output[i, j, 0] = gray_val
            output[i, j, 1] = gray_val
            output[i, j, 2] = gray_val
    
    return output

def simple_cvtColorBGRtoGrayBGR(image):
    return core_cvtColorBGRtoGrayBGR(image)


@jit(nopython=True, parallel=True, nogil=True)
def core_cvtColorBGRtoGray(image):
    h, w, _ = image.shape
    output = np.empty((h, w), dtype=np.uint8)
    
    for i in prange(h):
        for j in prange(w):
            gray_val = (image[i, j, 0] + image[i, j, 1] + image[i, j, 2]) // 3
            output[i, j] = gray_val
    
    return output

def simple_cvtColorBGRtoGray(image):
    return core_cvtColorBGRtoGray(image)



@jit(nopython=True, parallel=True, nogil=True)
def core_filter2D(padded_image, kernel):
    h, w = padded_image.shape[:2]
    k_height, k_width = kernel.shape
    pad_size = k_height // 2
    output = np.zeros((h - 2*pad_size, w - 2*pad_size))
    # loop unrolling to increase processing speed
    if k_height == 3:
        for i in prange(pad_size, h - pad_size):
            for j in prange(pad_size, w - pad_size):
                output[i - pad_size, j - pad_size] = (
                    padded_image[i-1, j-1] * kernel[0, 0] + 
                    padded_image[i-1, j] * kernel[0, 1] + 
                    padded_image[i-1, j+1] * kernel[0, 2] +
                    padded_image[i, j-1] * kernel[1, 0] + 
                    padded_image[i, j] * kernel[1, 1] + 
                    padded_image[i, j+1] * kernel[1, 2] +
                    padded_image[i+1, j-1] * kernel[2, 0] + 
                    padded_image[i+1, j] * kernel[2, 1] + 
                    padded_image[i+1, j+1] * kernel[2, 2]
                )
    elif k_height == 5:
        for i in prange(pad_size, h - pad_size):
            for j in prange(pad_size, w - pad_size):
                output[i - pad_size, j - pad_size] = (
                    padded_image[i-2, j-2] * kernel[0, 0] +
                    padded_image[i-2, j-1] * kernel[0, 1] +
                    padded_image[i-2, j] * kernel[0, 2] +
                    padded_image[i-2, j+1] * kernel[0, 3] +
                    padded_image[i-2, j+2] * kernel[0, 4] +
                    padded_image[i-1, j-2] * kernel[1, 0] +
                    padded_image[i-1, j-1] * kernel[1, 1] +
                    padded_image[i-1, j] * kernel[1, 2] +
                    padded_image[i-1, j+1] * kernel[1, 3] +
                    padded_image[i-1, j+2] * kernel[1, 4] +
                    padded_image[i, j-2] * kernel[2, 0] +
                    padded_image[i, j-1] * kernel[2, 1] +
                    padded_image[i, j] * kernel[2, 2] +
                    padded_image[i, j+1] * kernel[2, 3] +
                    padded_image[i, j+2] * kernel[2, 4] +
                    padded_image[i+1, j-2] * kernel[3, 0] +
                    padded_image[i+1, j-1] * kernel[3, 1] +
                    padded_image[i+1, j] * kernel[3, 2] +
                    padded_image[i+1, j+1] * kernel[3, 3] +
                    padded_image[i+1, j+2] * kernel[3, 4] +
                    padded_image[i+2, j-2] * kernel[4, 0] +
                    padded_image[i+2, j-1] * kernel[4, 1] +
                    padded_image[i+2, j] * kernel[4, 2] +
                    padded_image[i+2, j+1] * kernel[4, 3] +
                    padded_image[i+2, j+2] * kernel[4, 4]
                )
    else:
        for i in prange(pad_size, h - pad_size):
            for j in prange(pad_size, w - pad_size):
                sum_val = 0.0
                for k in range(-pad_size, pad_size+1):
                    for l in range(-pad_size, pad_size+1):
                        sum_val += padded_image[i+k, j+l] * kernel[pad_size+k, pad_size+l]
                output[i - pad_size, j - pad_size] = sum_val
    return output

def filter2D(image, kernel):
    image = image.astype(np.float32) / 255.0
    kernel = kernel.astype(np.float32)

    k_height, k_width = kernel.shape
    pad_size = k_height // 2

    if image.ndim == 3:
        h, w, c = image.shape
        output = np.zeros((h, w, c))
        for channel in prange(c):  # parallelizing over channels
            padded_image = np.pad(image[:, :, channel], ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
            output[:, :, channel] = core_filter2D(padded_image, kernel)
    else:
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
        output = core_filter2D(padded_image, kernel)
    
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output


@jit(nopython=True, parallel=True, nogil=True)
def histogram_parallel(image):
    n_threads = numba.config.NUMBA_NUM_THREADS
    hists = np.zeros((n_threads, 256), dtype=np.int64)
    
    n_pixels = image.size
    pixels_per_thread = n_pixels // n_threads
    
    for tid in prange(n_threads):
        start = tid * pixels_per_thread
        end = (tid + 1) * pixels_per_thread if tid != n_threads - 1 else n_pixels
        for idx in range(start, end):
            hists[tid, image.ravel()[idx]] += 1
            
    global_hist = hists.sum(axis=0)
    return global_hist

@jit(nopython=True, parallel=True, nogil=True)
def apply_lut_parallel(image, lut):
    h, w = image.shape
    output = np.empty_like(image)
    for i in prange(h):
        for j in prange(w):
            output[i, j] = lut[image[i, j]]
    return output

@jit(nopython=True)
def equalizeHist(image):
    hist = histogram_parallel(image)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    lut = ((cdf - cdf_min) * 255 / (image.size - cdf_min)).astype(np.uint8)
    return apply_lut_parallel(image, lut)
