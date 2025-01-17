"""
canny.py - Canny Edge detector
Reference: Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986
Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import generate_binary_structure, binary_erosion, label
#from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter as gaussian
from skimage import dtype_limits, img_as_float
from skimage._shared.utils import assert_nD


def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring masked pixels
    Parameters
    ----------
    image : array
        Image you want to smooth.
    function : callable
        A function that does image smoothing.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.
    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image

def canny(image, sigma=1., low_threshold=None, high_threshold=None, mask=None,
          use_quantiles=False,dx=1,dy=1):
    """Edge filter an image using the Canny algorithm.
    Parameters
    -----------
    image : 2D array
        Grayscale input image to detect edges on; can be of any dtype.
    sigma : float
        Standard deviation of the Gaussian filter.
    low_threshold : float
        Lower bound for hysteresis thresholding (linking edges).
        If None, low_threshold is set to 10% of dtype's max.
    high_threshold : float
        Upper bound for hysteresis thresholding (linking edges).
        If None, high_threshold is set to 20% of dtype's max.
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.
    use_quantiles : bool, optional
        If True then treat low_threshold and high_threshold as quantiles of the
        edge magnitude image, rather than absolute edge magnitude values. If True
        then the thresholds must be in the range [0, 1].
    Returns
    -------
    output : 2D array (image)
        The masked edge map. Values are edge angles (this has changed from the original)
    See also
    --------
    skimage.sobel
    Notes
    -----
    The steps of the algorithm are as follows:
    * Smooth the image using a Gaussian with ``sigma`` width.
    * Apply the horizontal and vertical Sobel operators to get the gradients
      within the image. The edge strength is the norm of the gradient.
    * Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.
    * Perform a hysteresis thresholding: first label all points above the
      high threshold as edges. Then recursively label any point above the
      low threshold that is 8-connected to a labeled point as an edge.
    References
    -----------
    .. [1] Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
           Pattern Analysis and Machine Intelligence, 8:679-714, 1986
    .. [2] William Green's Canny tutorial
           http://dasl.unlv.edu/daslDrexel/alumni/bGreen/www.pages.drexel.edu/_weg22/can_tut.html
    Examples
    --------
    >>> from skimage import feature
    >>> # Generate noisy image of a square
    >>> im = np.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>> im += 0.2 * np.random.rand(*im.shape)
    >>> # First trial with the Canny filter, with the default smoothing
    >>> edges1 = feature.canny(im)
    >>> # Increase the smoothing for better results
    >>> edges2 = feature.canny(im, sigma=3)
    """

    #
    # The steps involved:
    #
    # * Smooth using the Gaussian with sigma above.
    #
    # * Apply the horizontal and vertical Sobel operators to get the gradients
    #   within the image. The edge strength is the sum of the magnitudes
    #   of the gradients in each direction.
    #
    # * Find the normal to the edge at each point using the arctangent of the
    #   ratio of the Y sobel over the X sobel - pragmatically, we can
    #   look at the signs of X and Y and the relative magnitude of X vs Y
    #   to sort the points into 4 categories: horizontal, vertical,
    #   diagonal and antidiagonal.
    #
    # * Look in the normal and reverse directions to see if the values
    #   in either of those directions are greater than the point in question.
    #   Use interpolation to get a mix of points instead of picking the one
    #   that's the closest to the normal.
    #
    # * Label all points above the high threshold as edges.
    # * Recursively label any point above the low threshold that is 8-connected
    #   to a labeled point as an edge.
    #
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    #
    ##########################################################
    #   MODIFICATION BY EMMA HOWARD                          #
    ##########################################################
    #
    # - Scale the isobel and jsobel derivatives by input variables 
    #     dx and dy. This takes into account uneven grid-spacing
    # - Calculate the angle of the Canny edge and return masked angle array,
    #     instead of binary array
    #
    ##########################################################

    assert_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1 * dtype_max
    else:
        low_threshold = low_threshold / dtype_max

    if high_threshold is None:
        high_threshold = 0.2 * dtype_max
    else:
        high_threshold = high_threshold / dtype_max

    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    def fsmooth(x):
        return img_as_float(gaussian(x, sigma, mode='constant'))

    smoothed = smooth_with_function_and_mask(image, fsmooth, mask)
#    jsobel = ndi.sobel(smoothed, axis=1)     #EH CHANGE
    jsobel = ndi.sobel(smoothed, axis=1)/dx
#    isobel = ndi.sobel(smoothed, axis=0)     #EH CHANGE
    isobel = ndi.sobel(smoothed, axis=0)/dy
    theta = np.arctan2(jsobel,-isobel)         #EH ADD
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)
    magnitude = np.hypot(isobel, jsobel)

    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    s = generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)
    #
    #--------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(image.shape, bool)
    #----- 0 to 45 degrees ------
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    #
    #---- If use_quantiles is set then calculate the thresholds to use
    #
    if use_quantiles:
        if high_threshold > 1.0 or low_threshold > 1.0:
            raise ValueError("Quantile thresholds must not be > 1.0")
        if high_threshold < 0.0 or low_threshold < 0.0:
            raise ValueError("Quantile thresholds must not be < 0.0")

        high_threshold = np.percentile(magnitude, 100.0 * high_threshold)
        low_threshold = np.percentile(magnitude, 100.0 * low_threshold)

    #
    #---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)

    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                             np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    points = np.ma.masked_array(theta,mask=1-output_mask).filled(-999) # EH CHANGE
    return points  # EH CHANGE


def canny_div(div,lon2,lat2,low_threshold=2e-5,high_threshold=2e-5,window=5):
  masked,n=label(div>low_threshold)
  theta=-5*np.ones(div.shape)
  nmasked=np.array([(masked==i).sum() for i in range(n+1)])
  for i in range(window,div.shape[0]-window-1):
    for j in range(window,div.shape[1]-window-1):
      if div[i,j] > high_threshold and nmasked[masked[i,j]]>100:
        xc = ((masked[i-window:i+window,j-window:j+window] == masked[i,j])*lon2[i-window:i+window,j-window:j+window]).sum()/(masked[i-window:i+window,j-window:j+window] == masked[i,j]).sum()
        yc = ((masked[i-window:i+window,j-window:j+window] == masked[i,j])*lat2[i-window:i+window,j-window:j+window]).sum()/(masked[i-window:i+window,j-window:j+window] == masked[i,j]).sum()
        div_m = div[i-window:i+window,j-window:j+window]*(masked[i-window:i+window,j-window:j+window] == masked[i,j])
        ixx=(div_m*(lon2[i-window:i+window,j-window:j+window]-xc)**2).mean()
        ixy=(div_m*(lon2[i-window:i+window,j-window:j+window]-xc)*(lat2[i-window:i+window,j-window:j+window]-yc)).mean()
        iyy=(div_m*(lat2[i-window:i+window,j-window:j+window]-yc)**2).mean()
        m = np.array([[ixx,ixy],[ixy,iyy]])
        w,v=np.linalg.eig(m)
        if w[1]<w[0]:
          v = v[:,[1,0]]
          w = w[[1,0]]
        # v[:,1] has the smaller eigenvalue
        # v[:,0] has the bigger eigenvalue 
#        theta[i,j] = (np.arctan2(v[0,0],v[1,0])+np.pi/2)%np.pi-np.pi/2
        theta_tmp = np.arctan2(v[0,1],v[1,1])
        theta_tmp = theta_tmp%np.pi-np.pi/2.0
        a=-np.sin(theta_tmp)
        b=np.cos(theta_tmp)
        if 0 <= theta_tmp <=np.pi/4:
          dx_m = div[i,j+1]-div[i,j]
          dx_p = div[i,j]-div[i,j-1]
          dc_m = div[i+1,j-1]-div[i,j]
          dc_p = div[i,j]-div[i-1,j+1]
          d_m = (a+b)*dx_m - b*dc_m
          d_p = (a+b)*dx_p - b*dc_p
        elif np.pi/4 <= theta_tmp < np.pi/2:
          dy_m = div[i+1,j]-div[i,j]
          dy_p = div[i,j]-div[i-1,j]
          dc_m = div[i+1,j-1]-div[i,j]
          dc_p = div[i,j]-div[i-1,j+1]
          d_m = (a+b)*dy_m + a*dc_m
          d_p = (a+b)*dy_p + a*dc_p
        elif -np.pi/4 <= theta_tmp < 0:
          dy_m = div[i+1,j]-div[i,j]
          dy_p = div[i,j]-div[i-1,j]
          dd_m = div[i+1,j+1]-div[i,j]
          dd_p = div[i,j]-div[i-1,j-1]
          d_m = (b-a)*dy_m + a*dd_m
          d_p = (b-a)*dy_p + a*dd_p
        elif -np.pi/2 <= theta_tmp < -np.pi/4:
          dx_m = div[i,j+1]-div[i,j]
          dx_p = div[i,j]-div[i,j-1]
          dd_m = div[i+1,j+1]-div[i,j]
          dd_p = div[i,j]-div[i-1,j-1]
          d_m = (b-a)*dx_m + b*dd_m
          d_p = (b-a)*dx_p + b*dd_p
        if (d_m*d_p)<0:
          theta[i,j]=theta_tmp
  theta=np.ma.masked_array(theta,mask=(theta==-5))#%np.pi
  return theta


