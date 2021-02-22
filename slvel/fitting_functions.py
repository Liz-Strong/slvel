import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import pearsonr as pearsonr
"""Fitting functions for multi-Gaussian fitting.
"""
def fit_wrapper(x,*args):
    """
    This wrapper sets up the variables for the fit function.
    It allows for a variable numbers of Gaussians to be fitted.
    Calls multi_gaussian_fit_function
    
    Args:
        x (array):  x is independent variable x, such that y=f(x). 
        args: variable length argument list. args[0:n_gauss] are the amplitudes of the gaussians to be fitted. args[n_gauss:2*n_gauss] are the horizontal offsets of the gaussians to be fitted. args[2*n_gauss:3*n_gauss] are the standard deviations of the gaussians to be fitted. args[-1] is the vertical offset parameter
            
    Returns 
        multi_gaussian_fit_function(x,h,mu,sigma,vertical_offset)
    
    """
    n_gauss = (len(args)-1)//3 # number of gaussians that we're fitting
    
    h = args[0:n_gauss]
    mu = args[n_gauss:2*n_gauss]
    sigma = args[2*n_gauss:3*n_gauss]
    vertical_offset = args[-1]
    return multi_gaussian_fit_function(x,h,mu,sigma,vertical_offset)

def multi_gaussian_fit_function(x,h,mu,sigma,vertical_offset):
    """
    Returns a function that is comprised of an offset h and the 
    sum of gaussians with variable amplitudes, offsets, and standard 
    deviations (widths)
    
    Args:
        x (array): independent variable, such that y=f(x). 
        h (list): initial guesses for the amplitudes of the gaussians
        mu (list): initial guesses for the translational offsets of the gaussians
        sigma (list): initial guesses for standard deviations of gaussians
        vertical_offset (list): initial guess for vertical offset h
    
    Returns:
        fit (array): a function which consists of the sum of multiple gaussians and a vertical offset
    """
    # fit function starts with  vertical offset
    fit = np.zeros(len(x)) + vertical_offset
    
    # iterate through each amplitude/translational offset/standard deviation set & add them to the fit function
    for amp,offset,std in zip(h,mu,sigma):
        fit += amp*np.exp( -(x-offset)**2 / (2*std**2) )
    
    return fit

def initial_guess(initial_amplitude,initial_translational_offset,initial_stddev,initial_vertical_offset):
    """
    Create array with amplitude, phase and initial offset to be used in the curve fit
    
    Args:
        initial_amplitude (array): guess for the initial values of the amplitudes of the gaussians
        initial_translational_offset (array): guess for the initial values of the translational offsets of the gaussians
        initial_stddev (array): guess for the initial values of the standard deviations of the gaussians 
        initial_vertical_offset (float): guess for the initial values of the vertical offset 
    
    Returns:
        p0 (array): lists the initial_amplitude, initial_translational_offset, initial_stddev, initial_vertical_offset in the correct format for the curve fit.
    """
    #p0=[]
    #for a,mu,stddev in zip(initial_amplitude,initial_translational_offset,initial_stddev):
    #    p0.append([a,mu,stddev])
    #p0.append(initial_vertical_offset)
    
    p0 = [i for i in initial_amplitude]\
         + [i for i in initial_translational_offset]\
         + [i for i in initial_stddev]\
         + [initial_vertical_offset]
    
    return p0

def bound_maker(amplitude_bounds,translational_offset_bounds,stddev_bounds,vertical_offset_bounds,number_gaussians):
    """
    Create tuple with lower and upper bounds to be used in the curve fit
    
    Args:
        amplitude_bounds (tuple): bounds on the amplitudes of the gaussians
        translational_offset_bounds (tuple): bounds on the translational offsets of the gaussians
        stddev_bounds (tuple): bounds on the standard deviations of the gaussians
        vertical_offset_bounds (tuple): bounds on the vertical offset of the gaussians
        number_gaussians (int): the number of gaussians in the fit
    
    Returns:
        bounds (tuple): lists the bounds on the parameters used in the multigaussian fits
    """
    lower = [amplitude_bounds[0]]*number_gaussians + [translational_offset_bounds[0]]*number_gaussians + [stddev_bounds[0]]*number_gaussians + [vertical_offset_bounds[0]]
    upper = [amplitude_bounds[1]]*number_gaussians + [translational_offset_bounds[1]]*number_gaussians + [stddev_bounds[1]]*number_gaussians + [vertical_offset_bounds[1]]       
    bounds = (lower, upper)
    return bounds


def bound_maker_subsequent(t, y, y_fit, popt, num_new_gaussians, amplitude_bounds, stddev_bounds, vertical_offset_bounds, new_translational_offset, noise_avg):
    """
    Makes the bounds vector for fits after the first. Takes into account the previous fitted values
    
    Args:
        t (array): time grid of burst
        y (array): burst
        y_fit (array): previous fit to the burst
        popt (arrapy): the results from the multi gaussian curve fit of the previous fit
        num_new_gaussians (int): the number of gaussians to be added to the new fit
        amplitude_bounds (array): bounds on the amplitudes of the gaussians
        stddev_bounds (array): bounds on the standard deviations of the gaussians
        vertical_offset_bounds (array): bounds on the vertical offset
        new_translational_offset (tuple): 
    
    Returns:
        bounds (tuple): lists the bounds on the parameters used in the multigaussian fits
    """
    num_gaussians_old = int((len(popt)-1)/3)
    amplitudes = popt[:num_gaussians_old]
    translational_offsets = popt[num_gaussians_old:2*num_gaussians_old]
    widths = popt[2*num_gaussians_old:3*num_gaussians_old]
    vert_offset = popt[-1]
    
    lower_amp = np.append(amplitudes-np.abs(amplitudes)*.2, [-.75*(np.max(y)-noise_avg)]*num_new_gaussians)
    upper_amp = np.append(amplitudes+np.abs(amplitudes)*.2, [1.2*(np.max(y)-noise_avg)]*num_new_gaussians)

        
    # limit the movement of the previously fitted gaussians.
    lower_translational = np.append(translational_offsets*.8, [0]*num_new_gaussians)
    upper_translational = np.append(translational_offsets*1.2, [np.max(t)]*num_new_gaussians)
    if num_new_gaussians == 1:
        lower_translational = np.append(translational_offsets*.8, [new_translational_offset[-1]*.5])
        upper_translational = np.append(translational_offsets*1.2, [new_translational_offset[-1]*1.5])
    
    lower_translational[lower_translational<0] = 0
    upper_translational[upper_translational>np.max(t)] = .9*np.max(t)
    
    lower_stddev = np.append([stddev_bounds[0]]*num_gaussians_old, [stddev_bounds[0]]*num_new_gaussians)
    upper_stddev = np.append([stddev_bounds[1]]*num_gaussians_old, [stddev_bounds[1]]*num_new_gaussians)

    # make into array
    lower = np.concatenate((lower_amp, lower_translational, lower_stddev, [vertical_offset_bounds[0]]))
    upper = np.concatenate((upper_amp, upper_translational, upper_stddev, [vertical_offset_bounds[1]]))
    
    bounds = (lower, upper)
    return bounds

def calculate_r2(y, y_fit):
    """
    Calculates r2, the percentage of variability of the dependent variable that's 
    been accounted for. (how well the regression predicts the data)
    
    Args:
        y (array): data 
        yfit (array): is the fit to the data, evaluated using the same time axis as y
        
    Returns:
        r2 (float): characterizes how well y_fit predicts y
    """
    
    #ss_res = np.sum((y-y_fit)**2) #residual sum of squares
    #ss_tot = np.sum((y-np.mean(y))**2) #total sum of squares
    #r2 = 1-(ss_res/ss_tot) #r squared
    r2 = pearsonr(y, y_fit)[0]
    return r2

def calculate_rmse(targets, predictions):
    """
    Calculates root mean square error (RMSE) between targets and predictions
    
    Args:
        targets (array): actual values
        predictions (array): predicted values
        
    Returns:
        rmse (float): root mean square error
    """
    n = len(predictions)
    return np.linalg.norm(predictions - targets) / np.sqrt(n)

def calculate_max_error(targets, predictions):
    """
    Returns maximum absolute value of difference between target and predictions.
    
    Args:
        targets (array): actual values
        predictions (array): predicted values
        
    Returns:
        rmse (float): root mean square error
    
    """
    return np.max(np.abs(targets-predictions))

def gaussian_generator(npoints,std):
    """
    Make a gaussian f npoints long with standard deviation std
    
    Args:
        npoints (int): length of Gaussian
        std (float): standard deviation of Gaussian
        
    Returns:
        g (array): Gaussian 
    """
    g = signal.gaussian(npoints,std=std)
    return g

def rect_generator(npoints,width,area):
    """
    Make rect for correlation that has a height such that the height*width=area.
    
    Args:
        npoints (int): length of Gaussian
        width (float): width of rect
        area (float): area of rect. Dictates rect height via height = area/width
        
    Returns:
        r (array): rect function
    """
    r = np.zeros(npoints)
    r[int(np.floor(npoints/2-width/2+1)):int(np.ceil(npoints/2+width/2))] = area/(np.floor(width/2)*2) # do this flooring thing because the width gets rounded and we want the area constant always. 
    return r

def seed_initial_offsets_peaks(y, noise, rect_area=500, prominence_knockdown_factor=0.03):
    """
    Generate the locations of the seeds for the initial fit. Place a seed at each of the peaks.
    Determine peak location from smoothed version of signal. Smooth signal by cross correlating it with rect.
    
    Args:
        y (array): signal
        noise (float): noise level of y
        rect_area (float): area of rect, where area=width * height
        prominence_knockdown_factor (float): used to set the prominence for find_peaks as a function of the max height of xc_r
    
    Returns:
        peaks (array): list of the initial peaks
    
    """
    max_snr = np.max(y/noise)
    if max_snr>10:
        pass
    elif max_snr<5:
        prominence_knockdown_factor = .09
    else:
        prominence_knockdown_factor = .06
    
    length = len(y)
    r = rect_generator(length,length/35, rect_area)
    xc_r = signal.correlate(y,r)[length//2:-length//2] # cross correlation of signal and rect
    peaks, _ = find_peaks(xc_r, prominence=(np.max(xc_r)-noise*rect_area)*prominence_knockdown_factor)
    
    
    #plt.figure()
    #plt.plot(y, label='y')
    #plt.plot(xc_r/100, label='resid')
    #plt.plot(peaks, xc_r[15]/100*np.ones(len(peaks)), 'kx')
    #print(peaks)
    #plt.legend()
    
    return peaks

def initial_seed(t, y, noise,  max_num_gaussians=8, rect_area=500):
    """
    Makes seeds for the first fit.
    Calls seed_initial_offsets_peaks
    
    Args:
        t (array): time corresponding to signal
        y (array): signal
        noise (float): noise level of y
        max_num_gaussians (int): the maximum number of initial seeds
        rect_area (float): area of rect, where area=width * height
        prominence_knockdown_factor (float): used to set the prominence for find_peaks as a function of the max height of xc_r
    
    Returns:
        initial_translational_offset (array): a list of the initial conditions for the horizontal offsets, mu
        initial_amplitude (array): a list of the initial conditions for the amplitudes, A
    """
    peak_locns = seed_initial_offsets_peaks(y, noise, rect_area=rect_area) # use as initial mus
    peak_values = y[peak_locns] # use as initial amplitudes
    
    #plt.figure()
    #plt.plot(y)
    #plt.plot(peak_locns, peak_values, 'x')
    
    if len(peak_values)>max_num_gaussians:
        sorted_values = np.argsort(peak_values)[:max_num_gaussians]
        peak_values = peak_values[sorted_values]
        peak_locns = peak_locns[sorted_values]
        
    
    initial_translational_offset = t[peak_locns]
    initial_amplitude = peak_values-noise
    
    #because we subtract the noise from the initial amplitudes, some might be negative. get rid of those. 
    positive_value_locations = np.argwhere(initial_amplitude>0)
    initial_amplitude = initial_amplitude[positive_value_locations].flatten()
    initial_translational_offset = initial_translational_offset[positive_value_locations].flatten()
        
    return initial_translational_offset, initial_amplitude

def calculate_effective_length(model, fitted_vert_offset, delta_t, max_normalized_height=1):
    """
    Effective length is area divided by max height. 
    Here, this is the length of a rectangle with same max height as the signal
    
    Args:
        model (array): signal
        fitted_vert_offset (float): h in the multigaussian fitting equation
        delta_t (float): time discretization
        max_normalized_height (float): maximum height of the signal
    
    Returns:
        effective_length (float): effective length
    """
    area = np.sum(model-fitted_vert_offset)
    effective_length = area/(np.max(model)-fitted_vert_offset)*max_normalized_height*delta_t
    return effective_length

def calculate_burst_duration(y_fit, fitted_vert_offset, delta_t, lower_thresh=0.1, upper_thresh=0.9):
    """
    calculate the duration of the burst between the lower and upper 
    thresholds of the cumulative sum of the signal
    
    Args:
        y_fit (array): values of fitted burst
        fitted_vert_offset (float): h in the multigaussian fitting equation 
        delta_t (float): time discretization
        lower_thresh (float): lower fraction of signal to include in calculation
        upper_thresh (float): upper fraction of signal to include in calculation
    
    Returns:
        duration (float): time of signal between indices set by lower_thresh and upper_thresh operating on the integrated area of the signal
    """
    try:
        cs = np.cumsum(y_fit-fitted_vert_offset)
        csm = np.max(cs)
        lower_index = np.argwhere(cs>(lower_thresh*csm))[0]
        upper_index = np.argwhere(cs<(upper_thresh*csm))[-1]
        duration = (upper_index-lower_index) * delta_t
    except:
        print("problem calculating the duration")
        duration = [0]
    return duration[0]

def make_weights(y, g_length=100):
    """
    Makes the weighting function for the curve fitting operation. 
    Weights the signal to bias its larger magnitude components and to diminish the effect of the small components (i.e. the tails)
    Generates the weights from a smoothed copy of the burst, where this smoothed copy is made by cross correlating the signal with a gaussian.
    
    Args:
        y (array): signal
        g_length (int): length of Gaussian
        
    Returns:
        sigma (array): weights for the curve fitting scheme
    """
     # make weights
    length = len(y)
    g = gaussian_generator(length,g_length)
    xc_g = signal.correlate(y,g)[int(np.ceil(length/2-1)):-int(np.floor(length/2))]#[int(np.ceil(length/2)):-int(np.ceil(length/2))]
    weight = xc_g/np.max(xc_g)
    sigma = 1/np.sqrt(weight)
    return sigma

def eval_fit(y, y_fit, t, popt, delta_t):
    """
    Calculates metrics which characterize the efficacy of the fit.
    
    Args:
        y (array): signal
        y_fit (array): fitted version of the signal
        t (array): times corresponding to y and y_fit
        popt (array): results of curve fit
        delta_t float): time discretization
        
    Returns:
        r2 (float): percentage of variability of the burst that's been accounted for in the fit. (how well the regression predicts the data)
        rmse (float): root mean square error between fit and signal
        max_error (float): maximum absolute value of difference between y and y_fit
        max_error_normalized (float): max_error/max(y)
        duration (float): time of signal between indices set by lower_thresh and upper_thresh operating on the integrated area of the signal
    """
    fitted_vert_offset = popt[-1]
    
    #delta_t = t[1]-t[0]
    # calculate r^2
    r2 = calculate_r2(y,y_fit)

    # calculate rmse
    rmse = calculate_rmse(y,y_fit)

    # calculate max error
    max_error = calculate_max_error(y,y_fit)
    max_error_normalized = max_error/np.max(y)
    
    # calculate duration of burst in the middle 80% of it
    duration = calculate_burst_duration(y_fit, fitted_vert_offset, delta_t)
    
    return r2, rmse, max_error, max_error_normalized, duration

def package_fit_data(r2, rmse, max_error, max_error_normalized, 
                     time_eff, duration, popt, ang_vel, 
                     orbit_radius, x_offsets, y_offsets, number_gaussians, 
                     y, noise_avg, max_num_gaussians, dirctn, initial_number_gaussians, t):
    """
    Save data from fits for use in later modules.
    
    Args:
        r2 (float): percentage of variability of the burst that's been accounted for in the fit. (how well the regression predicts the data)
        rmse (float): root mean square error between fit and signal
        max_error (float): maximum absolute value of difference between y and y_fit
        max_error_normalized (float): max_error/max(y)
        time_eff (float): rect effective time of signal. time of a rectangle with same max height as the signal
        duration (float): time of signal between indices set by lower and upper 10% of the integrated area of signal
        popt (array): output of curve fit
        ang_vel (float): actual angular velocity Omega
        orbit_radius (float): actual orbit radius R
        x_offsets (float): actual x component of distance between orbit center and light center, D
        y_offsets (float): actual y component of distance between orbit center and light center, D
        number_gaussians (int): number of gaussians used to parameterize burst
        y (array): burst
        noise_avg (float): average noise value
        max_num_gaussians (int): maximum number of gaussians to be included in the fit
        dirctn (int): +1 or -1, clockwise or counter clockwise
        initial_number_gaussians (int): number of Gaussians used in initial fit
        t (array): time corresponding to y
        
    Returns:
        data (array): contains many of the arguments and several other metrics, packaged for pickling for use in next module.
    """
    
    try:
        # fit parameters 
        h = popt[0:number_gaussians] # amplitude
        mu = popt[number_gaussians:2*number_gaussians] # offset
        sigma = popt[2*number_gaussians:3*number_gaussians] # width

        # to save, we want distances relative to location of first gaussian
        sorted_indices = np.argsort(mu)
        #print('length of mu:', len(mu), 'length of popt:', len(popt), 'number of gaussians', number_gaussians,'length of h:', len(h))

        h_save, mu_save, sigma_save = np.zeros(max_num_gaussians), np.zeros(max_num_gaussians), np.zeros(max_num_gaussians)

        h_save[:number_gaussians] = h[sorted_indices]
        mu_save[:number_gaussians] = mu[sorted_indices]-mu[sorted_indices[0]] # subtract smalles from all to get relative offsets
        sigma_save[:number_gaussians] = sigma[sorted_indices]
        vert_offset_save = popt[-1]

        D = np.sqrt(x_offsets**2+y_offsets**2)
        theta = np.arctan2(y_offsets, x_offsets)

        max_SNR = np.max(y)/noise_avg
        avg_SNR = np.mean(y)/noise_avg
        
        if dirctn == 1:
            clockwise = [0]
            counterclockwise = [1]
        else:
            clockwise = [1]
            counterclockwise = [0]

        data = np.concatenate([[ang_vel],[rmse],
                               [r2],[max_error],[max_error_normalized],[time_eff],[duration],
                               h_save,mu_save,sigma_save,[vert_offset_save],[t[1]-t[0]],[orbit_radius], [x_offsets], 
                               [y_offsets], [D], [theta], [max_SNR], [avg_SNR], clockwise, counterclockwise, [int(initial_number_gaussians)]])

        return data
    except Exception as excptn:
        print("***\n***\nsomething went wrong in package_fit_data\n***\n***")
        print(excptn)
        return
    
def seed_later_offsets_peaks(y, noise, rect_area=100, prominence_knockdown_factor=0.03):
    """
    Makes seeds for fits following the first fit.
    
    Args:
        y (array): signal
        noise (float): noise level of y
        rect_area (float): area of rect, where area=width * height
        prominence_knockdown_factor (float): used to set the prominence for find_peaks as a function of the max height of xc_r
    
    Returns:
        peaks (array): a list of positions with which to seed the horizontal offsets, mu
    """

    length = len(y)
    r = rect_generator(length,length/25, rect_area)
    xc_r = signal.correlate(y,r)[length//2:-length//2]
    peaks, _ = find_peaks(xc_r, prominence=(np.max(xc_r)-noise*rect_area)*prominence_knockdown_factor)    
    """
    plt.figure()
    plt.plot(y, label='y')
    plt.plot(xc_r/100, label='resid')
    plt.legend()
    """
    return peaks

def subsequent_seeding(t, y, y_fit, popt, number_gaussians, noise_avg):
    """
    
    Args:
        t (array): times corresponding to y
        y (array): burst
        y_fit (array): fitted burst
        popt (arary): output of previous curve fit
        number_gaussians (array): number of gaussians in the fit
        noise_avg (float): average value of the noise
    
    Returns:
        new_translational_offset (array): initial guesses for the translational offsets of the gaussians
        new_amplitude (array): initial guesses for the amplitudes of the gaussians
    """
    residual = np.abs(y-y_fit)
    # find spot with largest residual; record its amplitude

    peaks = seed_later_offsets_peaks(residual, noise_avg)
    
    peak_to_use = peaks[np.argmax(residual[peaks])]

    """
    plt.figure(78)
    plt.plot(y, label='data')
    plt.plot(y_fit, label='fitted')
    plt.plot(residual, label="|residual|")
    plt.plot(peak_to_use, residual[peak_to_use], 'x')
    """
    new_gaussian_translational_offset = t[peak_to_use]
    new_gaussian_amplitude = residual[peak_to_use]
    
    #if new_gaussian_amplitude < 30:
    #    new_gaussian_amplitude = 100
    #new_translational_offset = np.append(initial_translational_offset, new_gaussian_translational_offset)
    
    # use the previously fitted peaks as initial conditions
    fitted_translational_offsets = popt[number_gaussians:number_gaussians*2]
    new_translational_offset = np.append(fitted_translational_offsets, new_gaussian_translational_offset)
    
    fitted_amplitudes = popt[:number_gaussians]
    new_amplitude = np.append(fitted_amplitudes, new_gaussian_amplitude)
    
    return new_translational_offset, new_amplitude

def fitting_function(selxn, t, y, noise_avg, noise_thresh, ang_vel, orbit_radius, x_offsets, y_offsets, dirctn, max_num_gaussians=8):
    """
    Performs multi Gaussian fits. Initializes first fit based on number of peaks in smoothed copy of burst. The residual of this fit is compared to the noise threshold. Until the absolute value of the residual is smaller than the noise threshold or until more than max_num_gaussians Gaussians are needed to parameterize the fit, subsequent fits place new Gaussians at locations which have large residuals. A great deal of care is taken in this function to standardize the weighting and initial conditions of the fits since the Gaussians inherently are not orthogonal. The goal is to produce fits with Gaussians which appear physical (aren't extremely tall and narrow or short and wide). The fits may not converge, or more gaussians than max_num_gaussians may be required to fit the function. In such cases, the fitting function passes the burst without returning a fit.
    
    Args:
        selxn (int): burst number being fitted
        t (array): times corresponding to y
        y (array): burst being fitted 
        noise_avg (float): average value of the noise
        noise_thresh (float): average value of the noise + standard deviation of noise
        ang_vel (float): actual angular velocity of underlying simulation Omega
        orbit_radius (float): actual orbit radius R
        x_offsets (float): actual x component of distance between orbit center and light center, D 
        y_offsets (float): actual y component of distance between orbit center and light center, D
        dirctn (int): +1 or -1 corresponding to direction of rotatoin
        max_num_gaussians (int): maximum number of gaussians used to fit the burst
        
    Returns:
        data (array): contains many of the arguments and several other metrics, packaged for pickling for use in next module.
    """
    # for initial fit, use peak finding to determine the number, 
    # location, and initaial amplitudes of the Gaussians. 
    initial_translational_offset, initial_amplitude = initial_seed(t, y, noise_avg)
    number_gaussians = len(initial_translational_offset)
    initial_number_gaussians = len(initial_translational_offset)
    
    if number_gaussians > max_num_gaussians:
        print("too many peaks were found initially: number_gaussians>max_number_gaussians.")
        return
    
    #calculate rect effective time to be used in the initial standard dev. condition.  
    delta_t = t[1]-t[0]
    time_eff = calculate_effective_length(y, noise_avg, delta_t) #instead of fitted_vert_offset, use noise_avg (we haven't yet fitted any fitted_vert_offset)
    #print("rect effective time: ", time_eff)
    initial_stddev_denominator = 1#np.random.randint(40, 60, 1)
    initial_stddev = [time_eff/9] * number_gaussians#[np.max(t)/initial_stddev_denominator] * number_gaussians
    initial_vertical_offset = noise_avg
    p0 = initial_guess(initial_amplitude,
                       initial_translational_offset,
                       initial_stddev,
                       initial_vertical_offset)

    #print("initial guesses: current time_eff-based stddev is ", initial_stddev[0], 'previous one was ', np.max(t)/50)
    # initialize curve fitting bounds
    amplitude_bounds = (0,np.max(y)-noise_avg*.25) 
    translational_offset_bounds = (0,np.max(t)) ### maybe make these somewhat closer to the seeds
    stddev_bounds = (np.max(t)/150,np.max(t)/2)
    vertical_offset_bounds = (2*noise_avg-noise_thresh,noise_thresh)# noise_thresh=mean+std, noise_avg=mean. so mean-std=2*noise_avg-noise_thresh
    bounds = bound_maker(amplitude_bounds,translational_offset_bounds,stddev_bounds,vertical_offset_bounds,number_gaussians)
    
    # make weights for fit
    sigma = make_weights(y)

    
    # limit the max number of function evaluations
    max_nfev = int(30*len(t))
    
    # try first fit
    try: 
        popt,pcov = curve_fit(lambda t,*p0:fit_wrapper(t,*p0),t,y,p0=p0,bounds=bounds,x_scale=np.max(t),sigma=sigma,max_nfev=max_nfev,absolute_sigma=False)
    except Exception as e:
        """plt.figure()
        plt.plot(t,y)"""
        print('p0:', p0)
        print('bounds', bounds)
        print('problem in first fit:', e)
        return
    ##### function will only reach this location if initial fit converged.
    
    # calculate residual 
    y_fit = fit_wrapper(t,*popt)
    residual = y-y_fit
    

    """
    plt.figure()
    plt.plot(t, y, label="data")
    plt.plot(t, y_fit, label="1st fit")
    plt.plot(t, np.abs(residual), label="|residual|")
    plt.plot([0, np.max(t)], [noise_thresh, noise_thresh], 'k--', label="threshold")
    plt.plot([0, np.max(t)], [noise_avg, noise_avg], 'k--', label="mean noise")

    plt.legend()
    """
    """
    print(noise_thresh)
    print(np.any(np.abs(residual)>noise_thresh))
    print(number_gaussians<max_num_gaussians)
    """
    
    # compare residual to noise threshold to determine whether or not 
    # another Gaussian should be added. Only add another Gaussian if 
    # there are no more than max_num_gaussians Gaussians already. 
    std_dev_residual_previous = np.std(y)
    std_dev_residual_new = np.std(residual)
    #print('std dev of residual is: ', std_dev_residual_new)
    while (np.any(np.abs(residual)>noise_thresh*1.1)) & (number_gaussians<max_num_gaussians) | (std_dev_residual_new<std_dev_residual_previous*.8):
        
        # try subsequent fit
        # add in another gausian 
        new_translational_offset, new_amplitude = subsequent_seeding(t, y, y_fit, popt, number_gaussians, noise_avg)
        
        old_stddev = popt[number_gaussians*2:number_gaussians*3]
        
        
        initial_stddev = np.append(old_stddev, time_eff/8)

        initial_vertical_offset = popt[-1]
        
        p0 = initial_guess(new_amplitude,
                           new_translational_offset,
                           initial_stddev,
                           initial_vertical_offset)
        # initialize curve fitting bounds
        num_new_gaussians = 1
        bounds = bound_maker_subsequent(t, y, y_fit, popt, num_new_gaussians, amplitude_bounds, stddev_bounds, vertical_offset_bounds, new_translational_offset, noise_avg)
        
        # try curve fit again

        try:
            popt,pcov = curve_fit(lambda t,*p0:fit_wrapper(t,*p0),t,y,p0=p0,bounds=bounds,x_scale=np.max(t),sigma=sigma,max_nfev=max_nfev,absolute_sigma=False)
        except: # if first fit fails to converge, end fitting
            print(selxn, "one of the subsequent fits failed to converge")
            return 
        y_fit = fit_wrapper(t,*popt)
        residual = y-y_fit
        number_gaussians += 1
        
        """
        plt.plot(t, y_fit, label="new fit")
        plt.plot(t, np.abs(residual), label="|new residual|")
        plt.legend()
        """
        
        std_dev_residual_previous = std_dev_residual_new
        std_dev_residual_new = np.std(residual)
        #print('std dev of residual is: ', std_dev_residual_new)
    
    if (np.any(np.abs(residual)<noise_thresh*1.1) & (number_gaussians<=max_num_gaussians)):
        print(selxn, "WORKED")
        # package data for ML input.
        r2, rmse, max_error, max_error_normalized, duration = eval_fit(y, y_fit, t, popt, delta_t)
        data = package_fit_data(r2, rmse, max_error, max_error_normalized, 
                                time_eff, duration, popt, ang_vel, 
                                orbit_radius, x_offsets, y_offsets, number_gaussians, y, 
                                noise_avg, max_num_gaussians, dirctn, initial_number_gaussians, t)

        return data
    else:
        print(selxn, "max number of gaussians reached, but fit not within noise threshold")
        return
    
    return

