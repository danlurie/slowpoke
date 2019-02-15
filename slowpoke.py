import os
import argparse
import numpy as np
import nibabel as nib
from scipy import signal as sci_signal
from nilearn import image, masking
from tqdm import tqdm

def get_numerator(signal_a, signal_b, lag):
    """
    Calculates the numerator of the cross-correlation equation.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
    lag : int
        Lag by which signal_b will be shifted relative to signal_a.
        
    Returns
    -------
    array_like (1D)
        Element-wise product of matching time points in the lagged signals.
    """
    if lag == 0:
        numerator = np.multiply(signal_a, signal_b)
    # If lag is positive, shift signal_b forwards relative to signal_a.
    if lag > 0:
        numerator = np.multiply(signal_a[lag:], signal_b[0:-lag])
    # If lag is negative, shift signal_b backward relative to signal_a.
    if lag < 0:
        numerator = np.multiply(signal_b[-lag:], signal_a[0:lag])
    return numerator

def get_denominator(signal_a, signal_b):
    """
    Calculates the denominator of the cross-correlation equation.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
        
    Returns
    -------
    float
        Product of the standard deviations of the input signals.
    """ 
    return np.std(signal_a) * np.std(signal_b)

def calc_xcorr(signal_a, signal_b, lag):
    """
    Calculate the cross-correlation of two signals at a given lag.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
    lag : int
        Lag by which signal_b will be shifted relative to signal_a.
        
    Returns
    -------
    float
        Normalized cross-correlation.
    """ 
    xcorr = np.true_divide(1., len(signal_a)-np.absolute(lag)) * np.sum(np.true_divide(get_numerator(signal_a, signal_b, lag),
              get_denominator(signal_a, signal_b)))
    return xcorr

def xcorr_range(signal_a, signal_b, lags):
    """
    Calculate the cross-correlation of two signals over a range of lags.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
    lags : array_like (1D)
        Lags by which signal_b will be shifted relative to signal_a.
        
    Returns
    -------
    array_like (1D)
        Normalized cross-correlation at each lag.
    """ 
    xcorr_vals = []
    for lag in lags:
        xcorr = calc_xcorr(signal_a, signal_b, lag)
        xcorr_vals.append(xcorr)
    return np.array(xcorr_vals)

# Adapted from https://gist.github.com/endolith/255291
def parabolic(sample_array, peak_index):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample local maximum when nearby samples are known.
   
    Parameters
    ----------
    sample_array : array_like (1D)
        Array of samples.
    peak_index : int
        Index for the local maximum in sample_array for which to estimate the inter-sample maximum.
   
    Returns
    -------
    tuple
        The (x,y) coordinates of the vertex of a parabola through peak_index and its two neighbors.
    """
    vertex_x = 1/2. * (sample_array[peak_index-1] - sample_array[peak_index+1]) / (sample_array[peak_index-1] - 2 * sample_array[peak_index] + sample_array[peak_index+1]) + peak_index
    vertex_y = sample_array[peak_index] - 1/4. * (sample_array[peak_index-1] - sample_array[peak_index+1]) * (vertex_x - peak_index)
    return (vertex_x, vertex_y)

def gen_lag_map(epi_img, brain_mask_img, gm_mask_img, lags):
    print("...masking data...")
    epi_gm_masked_data = masking.apply_mask(epi_img, gm_mask_img)
    gm_mean_signal = np.mean(epi_gm_masked_data, axis=1)
    epi_brain_masked_data = masking.apply_mask(epi_img, brain_mask_img).T
    lag_index_correction = np.sum(np.array(lags) > 0)
    xcorr_array = [] 
    print("...calculating lags...")
    for voxel in tqdm(epi_brain_masked_data, unit='voxel'):
        vox_signal = voxel
        vox_xcorr = xcorr_range(gm_mean_signal, vox_signal, lags)
        xcorr_maxima = sci_signal.argrelmax(np.array(vox_xcorr), order=1)[0]
        if len(xcorr_maxima) == 0:
            interp_max = np.argmax(vox_xcorr) - lag_index_correction
        elif len(xcorr_maxima) == 1:
            interp_max = parabolic(vox_xcorr, xcorr_maxima[0])[0]
            interp_max = interp_max - lag_index_correction
        elif len(xcorr_maxima) > 1:
            xpeak = xcorr_maxima[np.argmax(vox_xcorr[xcorr_maxima])]
            interp_max = parabolic(vox_xcorr, xpeak)[0]
            interp_max = interp_max - lag_index_correction
        xcorr_array.append(interp_max)
    
    return(masking.unmask(np.array(xcorr_array), brain_mask_img))

parser = argparse.ArgumentParser()
# Positional arguments
parser.add_argument("epi_path", help="Path to a 4D NIfTI image.", type=str)
parser.add_argument("brain_mask_path", help="Path to a brain mask. Must be the same space and resolution as the EPI image.", type=str)
parser.add_argument("gm_map_path", help="Path to a gray matter probability map. Must be in the same space as the EPI image (but does not need to be the same resolution).", type=str)
parser.add_argument("out_dir", help="Directory where slowpoke should output lag map images.")
parser.add_argument("out_prefix", help="Filename prefix for the lag map images created by slowpoke.")

# Optional arguments
parser.add_argument("--max_lag", help="Maximum lag (in TRs) at which cross-correlations should be calculated. The default is 20.", default=20, type=int)
parser.add_argument("--smoothing_kernel", help="Full-width at half maximum (in mm) of a Gaussian filter. If provided, slowpoke will output a smoothed version of the lag map in addition to the un-smoothed version.", type=float)
parser.add_argument("--gm_thresh", help="Value at which to threshold/binarize the gray matter probability map. The default is 0.9.", default=0.9, type=float)

# Get the command line arguments
args = parser.parse_args()

# Check required inputs
assert os.path.exists(args.epi_path), "The specified EPI does not exist!"
assert os.path.exists(args.brain_mask_path), "The specified brain mask does not exist!"
assert os.path.exists(args.gm_map_path), "The specified GM probability map does not exist!"
assert os.path.exists(args.out_dir), "The specified output directory does not exist!"

# Check optional inputs
assert args.gm_thresh > 0.0, "The GM probability threshold must be between 0.0 and 1.0!"
assert not (args.gm_thresh > 1.0), "The GM probability threshold must be between 0.0 and 1.0!"

print("Loading images...")
# Load the EPI image.
epi_img = nib.load(args.epi_path)
assert len(epi_img.shape) == 4, "The specified EPI image is not 4D!" 

# Load the brain mask.
brain_mask_img = nib.load(args.brain_mask_path)

# Check that the brain mask and EPI have the same 3D dimensions.
assert epi_img.shape[0:-1] == brain_mask_img.shape, "The EPI image and brain mask must be the same resolution!"

# Load the GM map.
gm_map_img = nib.load(args.gm_map_path)

print("Creating GM mask...")
# Resample the GM map to the resolution of the EPI image.
gm_map_resampled_img = image.resample_to_img(gm_map_img, brain_mask_img)

# Threshold and binarize the GM map to create a GM mask.
mask_eqn = "(img1 > {0}).astype(bool)".format(args.gm_thresh)
gm_mask_img = image.math_img(mask_eqn, img1=gm_map_resampled_img)

# Constrain the GM mask by the brain mask.
gm_mask_img = masking.intersect_masks([gm_mask_img, brain_mask_img])

print("Creating lag map...")
# Generate the lag map
lags = range(-args.max_lag, args.max_lag+1)
lag_map_img = gen_lag_map(epi_img, brain_mask_img, gm_mask_img, lags)

print("Saving lag map(s)....")
# Save the lag map.
lag_map_img.to_filename(os.path.join(args.out_dir, args.out_prefix+'_MaxLag-{0}_lags.nii.gz'.format(str(args.max_lag))))

# Smooth the lag map.
lag_map_smooth_img = image.smooth_img(lag_map_img, args.smoothing_kernel)

# Save the smoothed lag map.
lag_map_smooth_img.to_filename(os.path.join(args.out_dir, args.out_prefix+'_MaxLag-{0}_FWHM-{1}_lags.nii.gz'.format(str(args.max_lag), str(args.smoothing_kernel))))
print("Done! What took you so long, slowpoke?")

