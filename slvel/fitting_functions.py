import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import pearsonr as pearsonr
from scipy.special import erf as erf

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
    Make the seeds for fits after the first fit.
    
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




"""
**********************
Fitting functions for erf-rect-erfs (use these when features within the illuminating beam have top hat intensity profiles)
**********************
"""




def error_function(x, x0, w):
    """
    Error function with equation y=0.5*(1+erf(np.sqrt(2)*(x-x0)/w))
    
    Args:
        x: array of independent variable (x) values
        x0: error function offset
        w: error function width
    
    Retunrs:
        y: computed error function
    """
    y = 0.5*(1+erf(np.sqrt(2)*(x-x0)/w))
    return y

def error_function_complimentary(x, x0, w):
    """
    Complimentary error function with equation y=0.5*(1-erf(np.sqrt(2)*(x-x0)/w))
    
    Args:
        x: data x values
        x0: error function offset
        w: error function width
    
    Returns:
        y: computed error function
    """
    y = 0.5*(1-erf(np.sqrt(2)*(x-x0)/w))
    return y

def fit_wrapper_erfrecterf(x,*args):
    """ 
    This wrapper sets up the variables for the fit function.
    It allows for a variable numbers of erf-rect-erfs to be fitted.
    Calls erf_rect_fit_function
    
    Args:
        x (array):  x is independent variable x, such that y=f(x). 
        args: variable length argument list. args[0:n_erfs] are the amplitudes of the erf-rect-erfs to be fitted. Each erf-rect-erf feature has an erf and a complimentary erf. args[n_erfs:2*n_erfs] are the horizontal offsets of the erf to be fitted. args[2*n_erfs:3*n_erfs] are the widths of the erf to be fitted. args[3*n_erfs:4*n_erfs] and  args[4*n_erfs:5*n_erfs] are the horizontal offsets and widths of the complimentary erf to be fitted. args[-1] is the vertical offset parameter
            
    Returns 
        erf_rect_fit_function(x,a,mu0,sigma0,mu1,sigma1,vertical_offset)
    """
    n_erfs = (len(args)-1)//5 # number of erf-rect-erf features that we're fitting
    
    a = args[0:n_erfs]
    mu0 = args[n_erfs:2*n_erfs]
    sigma0 = args[2*n_erfs:3*n_erfs]
    mu1 = args[3*n_erfs:4*n_erfs]
    sigma1 = args[4*n_erfs:5*n_erfs]
    vertical_offset = args[-1]
    return erf_rect_fit_function(x, a, mu0, sigma0, mu1, sigma1, vertical_offset)

def erf_rect_fit_function(x,a,mu0,sigma0,mu1,sigma1,vertical_offset):
    """ 
    Returns a function that is comprised of erf-rect-erf features. Each feature has a top-hat profile 
    generated as the sum of an error function at one time and a complimentary error function at a later time. 
    
    Args:
        x (array): independent variable, such that y=f(x). 
        a (list): initial guesses for the amplitudes of the erf-rect-erfs
        mu0 (list): initial guesses for the translational offsets of the erfs
        sigma0 (list): initial guesses for standard deviations of erfs
        mu1 (list): initial guesses for the translational offsets of the complimentary erfs
        sigma1 (list): initial guesses for standard deviations of the complimentary erfs
        vertical_offset (list): initial guess for vertical offset h
    
    Returns:
        fit (array): a function which consists of the sum of multiple gaussians and a vertical offset
    
    
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
    # initialize fi function & add the vertical offset to the fit function
    fit = np.zeros(len(x))+vertical_offset
    
    # iterate through each erf-rect-erf and add it to the fit function
    for amp, offset0, std0, offset1, std1 in zip(a, mu0, sigma0, mu1, sigma1):
        fit += amp*(error_function(x, offset0, std0) + error_function_complimentary(x, offset1, std1) -1 )
     
    return fit

def initial_guess_erfrecterf(initial_amplitude,initial_translational_offset0,initial_stddev0,initial_translational_offset1,initial_stddev1,initial_vertical_offset):
     """
    Create array with amplitude, standard deviation, translational offsets, and vertical offset to be used in the curve fit
    
    Args:
        initial_amplitude (array): guess for the initial values of the amplitudes of the erf-rect-erf
        initial_translational_offset0 (array): guess for the initial values of the translational offsets of the erf
        initial_stddev0 (array): guess for the initial values of the standard deviations of the erf
        initial_translational_offset1 (array): guess for the initial values of the translational offsets of the complimentary erf
        initial_stddev1 (array): guess for the initial values of the standard deviations of the complimentary erf
        initial_vertical_offset (float): guess for the initial values of the vertical offset 
    
    Returns:
        p0 (array): lists the initial_amplitude, initial_translational_offset of the erf, initial_stddev of the erf, initial_translational_offset of the complimentary erf, initial_stddev of the complimentary erf, initial_vertical_offset in the correct format for the curve fit.
    """    
    p0 = [i for i in initial_amplitude]\
         + [i for i in initial_translational_offset0]\
         + [i for i in initial_stddev0]\
         + [i for i in initial_translational_offset1]\
         + [i for i in initial_stddev1]\
         + [initial_vertical_offset]
    
    return p0

def bound_maker_erfrecterf(amplitude_bounds,translational_offset_bounds,stddev_bounds,vertical_offset_bounds,number_erfs):
    """
    Create tuple with lower and upper bounds to be used in the curve fit
    
    Args:
        amplitude_bounds (tuple): bounds on the amplitudes of the gaussians
        translational_offset_bounds (tuple): bounds on the translational offsets of the gaussians
        stddev_bounds (tuple): bounds on the standard deviations of the gaussians
        vertical_offset_bounds (tuple): bounds on the vertical offset of the gaussians
        number_erfs (int): the number of erf-rect-erf features in the fit
    
    Returns:
        bounds (tuple): lists the bounds on the parameters used in the erf-rect-erf fits
    """
    lower = [amplitude_bounds[0]]*number_erfs + [translational_offset_bounds[0]]*number_erfs + [stddev_bounds[0]]*number_erfs + [translational_offset_bounds[0]]*number_erfs + [stddev_bounds[0]]*number_erfs + [vertical_offset_bounds[0]]
    upper = [amplitude_bounds[1]]*number_erfs + [translational_offset_bounds[1]]*number_erfs + [stddev_bounds[1]]*number_erfs + [translational_offset_bounds[1]]*number_erfs + [stddev_bounds[1]]*number_erfs + [vertical_offset_bounds[1]]       
    bounds = (lower, upper)
    return bounds

def bound_maker_subsequent_erfrecterf(t, y, y_fit, popt, num_new_erfs, amplitude_bounds, sigma0_bounds, sigma1_bounds, vertical_offset_bounds, new_mu0, new_mu1):
    """
    Makes the bounds vector for fits after the first. Takes into account the previous fitted values
    
    Args:
        t (array): time grid of burst
        y (array): burst
        y_fit (array): previous fit to the burst
        popt (arrapy): the results from the multi gaussian curve fit of the previous fit
        num_new_erfs (int): the number of gaussians to be added to the new fit
        amplitude_bounds (array): bounds on the amplitudes of the gaussians
        sigma0_bounds (array): bounds on the standard deviations of the erf
        sigma1_bounds (array): bounds on the standard deviations of the complimentary erf
        vertical_offset_bounds (array): bounds on the vertical offset
        new_mu0 (array): new value for the translational position of the  erf 
        new_mu1 (array): new value for the translational position of the complimentary erf
    
    Returns:
        bounds (tuple): lists the bounds on the parameters used in the erf-rect-erf fits
    """
    amplitudes = popt[0:num_erfs_old]
    mu0 = popt[num_erfs_old:2*num_erfs_old]
    sigma0 = popt[2*num_erfs_old:3*num_erfs_old]
    mu1 = popt[3*num_erfs_old:4*num_erfs_old]
    sigma1 = popt[4*num_erfs_old:5*num_erfs_old]
    vertical_offset = popt[-1]
    
    lower_amp = np.append(amplitudes-np.abs(amplitudes)*.2, [0]*num_new_erfs)
    upper_amp = np.append(amplitudes+np.abs(amplitudes)*.4, [1.2*(np.max(y)-noise_avg)]*num_new_erfs)

        
    # limit the movement of the previously fitted erf-rect-erfs.
    lower_mu0 = np.append(mu0*.8, [0]*num_new_erfs)
    upper_mu0 = np.append(mu0*1.2, [np.max(t)]*num_new_erfs)
    lower_mu1 = np.append(mu1*.8, [0]*num_new_erfs)
    upper_mu1 = np.append(mu1*1.2, [np.max(t)]*num_new_erfs)
    if num_new_erfs == 1:
        lower_mu0 = np.append(mu0*.8, [new_mu0[-1]*.5])
        upper_mu0 = np.append(mu0*1.2, [new_mu0[-1]*1.5])
        lower_mu1 = np.append(mu1*.8, [new_mu1[-1]*.5])
        upper_mu1 = np.append(mu1*1.2, [new_mu1[-1]*1.5])
    lower_mu0[lower_mu0<0] = 0
    lower_mu1[lower_mu1<0] = 0
    upper_mu0[upper_mu0>np.max(t)] = .9*np.max(t)
    upper_mu1[upper_mu1>np.max(t)] = .9*np.max(t)
    
    
    lower_sigma0 = np.append([sigma0_bounds[0]]*num_erfs_old, [sigma0_bounds[0]]*num_new_erfs)
    lower_sigma1 = np.append([sigma1_bounds[0]]*num_erfs_old, [sigma1_bounds[0]]*num_new_erfs)
    upper_sigma0 = np.append([sigma0_bounds[1]]*num_erfs_old, [sigma0_bounds[1]]*num_new_erfs)
    upper_sigma1 = np.append([sigma1_bounds[1]]*num_erfs_old, [sigma1_bounds[1]]*num_new_erfs)

    # make into array
    lower = np.concatenate((lower_amp, lower_mu0, lower_sigma0, lower_mu1, lower_sigma1, [vertical_offset_bounds[0]]))
    upper = np.concatenate((upper_amp, upper_mu0, upper_sigma0, upper_mu1, upper_sigma1, [vertical_offset_bounds[1]]))
    
    bounds = (lower, upper)
    return bounds

def find_edges(y, trigger_height):
    """
    Simple zero-crossing algorithm to locate the rising and falling edges of a signal. If the signal is noisy around the location of the threshold, then multiple rising and falling edges may be detected where only one should be detected. If this happens, try smoothing the signal beforehand or selecting only a single of the set of falsely identified edge positions.
    
    Args:
        y (array): signal of interest
        trigger_height (float): the height at which a rising or falling edge is detected
        
    Returns:
        potential_rising_edges (list): list of rising edges at which to seed erf-rect-erfs
        potential_falling_edges (list): list of falling edges at which to seed erf-rect-erfs
    """
    potential_falling_edge, potential_rising_edge  = [], []
    for num, (i,j) in enumerate(zip(y[:-1], y[1:])):
        if (i>trigger_height) and (j<trigger_height):
            potential_falling_edge.append(num)
        if (i< trigger_height) and (j>trigger_height):
            potential_rising_edge.append(num)
    return potential_rising_edge, potential_falling_edge

def seed_initial_offsets_edges(y, noise_level):
    """
    Seed the starts and the edges 
    
    Args:
        y (array): signal of interest
        noise_level (float): the mean noise level of the signal. The threshold for finding the edges is based on this value
        
    Returns:
        rising_edges (list): list of rising edges at which to seed erf-rect-erfs
        falling_edges (list): list of falling edges at which to seed erf-rect-erfs
    """
    
    threshold = noise_level*2
    rising_edges, falling_edges = find_edges(y, threshold)
    
    
    return rising_edges, falling_edges

def seed_initial_offsets_edges_smoothed(y, noise):
    """
    Seed the starts and the edges 
    
    Inputs:
        y (array): signal of interest
        noise (float): the mean noise level of the signal. The threshold for finding the edges is based on this value
        
    Returns:
        rising_edges (list): list of rising edges at which to seed erf-rect-erfs
        falling_edges (list): list of falling edges at which to seed erf-rect-erfs
    """
    
    # find the major peaks
    threshold = np.max(y)*.25
    
    # Find edges of smoothed signal 
    area = 4000
    length = len(y)//50
    width = len(y)
    r = rect_generator(length,width,area)
    xc_r = signal.correlate(y,r)[length//2:-length//2]
    normalized_xc_r = xc_r/np.max(xc_r)*np.max(y)
    
    rising_edges, falling_edges = find_edges(normalized_xc_r, threshold)
    
    """ plt.figure()
    plt.plot(y)
    plt.plot([0, len(y)], [threshold,threshold], 'm')
    print(rising_edges)"""
    return rising_edges, falling_edges, xc_r

def initial_seed_erfrecterf(t, y, noise):
    """
    Makes seeds for the first fit.
    Calls seed_initial_offsets_peaks
    
    Args:
        t (array): time corresponding to signal
        y (array): signal
        noise (float): noise level of y
        
    Returns:
        initial_translational_offset (array): a list of the initial conditions for the horizontal offsets, mu
        initial_amplitude (array): a list of the initial conditions for the amplitudes, A
    """
    rising_edges, falling_edges, xc_r = seed_initial_offsets_edges_smoothed(y, noise) 

    #initial_translational_offset = t[peak_locns]
    initial_amplitudes = []
    for r,f in zip(rising_edges, falling_edges):
        initial_amplitudes.append(y[int((f-r)//2+r)]-noise)
    #print("initial amplitudes:", initial_amplitudes)
    initial_amplitudes = np.asarray(initial_amplitudes)
    

    initial_mu0 = t[rising_edges]
    initial_mu1 = t[falling_edges]
        
    return initial_mu0, initial_mu1, initial_amplitudes

def seed_later_offsets_peaks_erfrecterf(y, noise_level):
    """
    Seed the starts and the edges of the erf-rect-erf features
    
    Args:
        y (array): signal of interest
        noise_level (float): the mean noise level of the signal. The threshold for finding the edges is based on this value
        
    Returns:
        rising_edge (int): rising edges at which to seed erf-rect-erfs which corresponds to the location of the largest residual
        falling_edge (int): falling edges at which to seed erf-rect-erfs which corresponds to the location of the largest residual
    """
    
    threshold = noise_level
    rising_edges, falling_edges = find_edges(y, threshold)
    
    # find the location with the lartest peak
    peak_val = []
    for r,f in zip(rising_edges, falling_edges):
        peak_val.append(np.abs(y[(f-r)//2+r]))
    if not peak_val: #if peak_val is empty
        threshold = noise_level*.5
        rising_edges, falling_edges = find_edges(y, threshold)
        for r,f in zip(rising_edges, falling_edges):
            peak_val.append(np.abs(y[(falling_edge-rising_edge)//2+rising_edge]))
    if not peak_val:
        return 
    else:
        biggest_residual_location = np.argmax(peak_val)
    return rising_edges[biggest_residual_location], falling_edges[biggest_residual_location]

def subsequent_seeding_erfrecterf(t, y, y_fit, popt, number_erfs, noise_threshold):
    """
    Make the seeds for fits after the first fit.
    
    Args:
        t (array): times corresponding to y
        y (array): burst
        y_fit (array): fitted burst
        popt (arary): output of previous curve fit
        number_erfs (array): number of erf-rect-erf features in the fit
        noise_avg (float): average value of the noise
    
    Returns:
        new_translational_offset (array): initial guesses for the translational offsets of the erf-rect-erf features
        new_amplitude (array): initial guesses for the amplitudes of the erf-rect-erf features
    """
    residual = np.abs(y-y_fit)
    plt.figure()
    plt.plot(residual)
    plt.plot(y)
    plt.plot(y_fit)
    
    # find spot with largest residual; record its amplitude
    try:
        rising_edge, falling_edge = seed_later_offsets_peaks_erfrecterf(residual, noise_threshold) 
        print(rising_edge, falling_edge)

        mu0_new = t[rising_edge]
        mu1_new = t[falling_edge]

        print('falling edge is ',falling_edge)
        a_new = y[(falling_edge-rising_edge)//2+rising_edge]-noise_threshold
        sigma0_new = 5 
        sigma1_new = 5
        
        """
        plt.figure(78)
        plt.plot(y, label='data')
        plt.plot(y_fit, label='fitted')
        plt.plot(residual, label="|residual|")
        plt.plot(peak_to_use, residual[peak_to_use], 'x')
        """

        # use the previously fitted peaks as initial conditions
        fitted_a = popt[:number_erfs]
        new_a = np.append(fitted_a, a_new)

        fitted_mu0 = popt[number_erfs:2*number_erfs]
        new_mu0 = np.append(fitted_mu0, mu0_new
                           )
        fitted_sigma0 = popt[2*number_erfs:3*number_erfs]
        new_sigma0 = np.append(fitted_sigma0, sigma0_new)

        fitted_mu1 = popt[3*number_erfs:4*number_erfs]
        new_mu1 = np.append(fitted_mu1, mu1_new)

        fitted_sigma1 = popt[4*number_erfs:5*number_erfs]
        new_sigma1 = np.append(fitted_sigma1, sigma1_new)
        
    except Exception as e: 
        print("Exception in subsequent_seeding", e)
        return 
    
    return new_a, new_mu0, new_sigma0, new_mu1, new_sigma1


def package_fit_data_erfrecterf(r2, rmse, max_error, max_error_normalized, 
                     time_eff, duration, popt, fr, 
                     number_erfrecterfs, 
                     y, noise_avg, max_num_erfrecterfs, initial_number_erfrecterfs):
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
        fr (float): flow rate
        number_erfrecterfs (int): number of erf-rect-erf features used to parameterize burst
        y (array): burst
        noise_avg (float): average noise value
        max_num_erfrecterfs (int): maximum number of gaussians to be included in the fit
        initial_number_erfrecterfs (int): number of Gaussians used in initial fit
        
    Returns:
        data (array): contains many of the arguments and several other metrics, packaged for pickling for use in next module.
    """
    
    try:
        # fit parameters 
        a = popt[0:number_erfrecterfs] # amplitude
        mu0 = popt[number_erfrecterfs:2*number_erfrecterfs] # offset
        sigma0 = popt[2*number_erfrecterfs:3*number_erfrecterfs] # width
        mu1 = popt[3*number_erfrecterfs:4*number_erfrecterfs] # offset
        sigma1 = popt[4*number_erfrecterfs:5*number_erfrecterfs] # width

        # to save, we want distances relative to location of first erfrecterf
        sorted_indices = np.argsort(mu0)
        #print('length of mu:', len(mu), 'length of popt:', len(popt), 'number of gaussians', number_gaussians,'length of h:', len(h))

        a_save, mu0_save, sigma0_save, mu1_save, sigma1_save = np.zeros(max_num_erfrecterfs), np.zeros(max_num_erfrecterfs), np.zeros(max_num_erfrecterfs), np.zeros(max_num_erfrecterfs), np.zeros(max_num_erfrecterfs)

        a_save[:number_erfrecterfs] = a[sorted_indices]
        mu0_save[:number_erfrecterfs] = mu0[sorted_indices]-mu0[sorted_indices[0]] # subtract smalles from all to get relative offsets
        sigma0_save[:number_erfrecterfs] = sigma0[sorted_indices]
        mu1_save[:number_erfrecterfs] = mu1[sorted_indices]-mu0[sorted_indices[0]] # subtract smalles from all to get relative offsets
        sigma1_save[:number_erfrecterfs] = sigma1[sorted_indices]
        vert_offset_save = popt[-1]

        max_SNR = np.max(y)/noise_avg
        avg_SNR = np.mean(y)/noise_avg
        
        data = np.concatenate([[fr],[rmse],
                               [r2],[max_error],[max_error_normalized],[time_eff],[duration],
                               a_save,mu0_save,sigma0_save,mu1_save, sigma1_save,[vert_offset_save],[t[1]-t[0]],
                               [max_SNR], [avg_SNR], [int(initial_number_erfrecterfs)]])

        return data
    except Exception as e:
        print('Exception:', e)
        print("***\n***\nsomething went wrong in package_fit_data\n***\n***")
        return
    
def fitting_function_erfrecterf(selxn, t, y, noise_avg, noise_thresh, fr, max_num_erfrecterfs=4):
    """
    Performs erf-rect-erf fits. Initializes first fit based on number of edges in smoothed copy of burst. The residual of this fit is compared to the noise threshold. Until the absolute value of the residual is smaller than the noise threshold or until more than max_num_erfrecterfs features are needed to parameterize the fit, subsequent fits place new Gaussians at locations which have large residuals. A great deal of care is taken in this function to standardize the weighting and initial conditions of the fits since the erf-rect-erf features inherently are not orthogonal. The goal is to produce fits with Gaussians which appear physical (aren't extremely tall and narrow or short and wide). The fits may not converge, or more features than max_num_erfrecterfs may be required to fit the function. In such cases, the fitting function passes the burst without returning a fit.
    
    Args:
        selxn (int): burst number being fitted
        t (array): times corresponding to y
        y (array): burst being fitted 
        noise_avg (float): average value of the noise
        noise_thresh (float): average value of the noise + standard deviation of noise
        fr (float): actual flow rate underlying simulation 
        max_num_erfrecterfs (int): maximum number of erf-rect-erf features used to fit the burst
        
    Returns:
        data (array): contains many of the arguments and several other metrics, packaged for pickling for use in next module.
    """
    
    # check that there are enough points above the noise threshold to actually do a fit
    if np.shape(np.argwhere(y>noise_avg + 3*(noise_thresh-noise_avg)))[0]<12:
        print("not enough of the burst has an intensity greater than 2x the noise threshold ")
        return
    
    # for initial fit, use peak finding to determine the number, 
    # location, and initaial amplitudes of the Gaussians. 
    initial_mu0, initial_mu1, initial_amplitude = initial_seed_erfrecterf(t, y, noise_thresh)
    number_erfrecterfs = len(initial_mu0)

    
    if number_erfrecterfs > max_num_erfrecterfs:
        print("too many peaks were found initially: number_erfrecterfs > max_num_erfrecterfs.")
        return
    
    #calculate rect effective time to be used in the initial standard dev. condition.  
    delta_t = t[1]-t[0]
    time_eff = calculate_effective_length(y, noise_avg, delta_t) #instead of fitted_vert_offset, use noise_avg (we haven't yet fitted any fitted_vert_offset)
    #print("rect effective time: ", time_eff)
    initial_sigma0 = [time_eff/5]*number_erfrecterfs
    initial_sigma1 = [time_eff/5]*number_erfrecterfs
    
    # initialize vertical offset
    initial_vertical_offset = noise_avg + np.mean( [np.mean(y[:len(y)//5]), np.mean(y[4*len(y)//5:])] ) 

    
    p0 = initial_guess(initial_amplitude,
                       initial_mu0,
                       initial_sigma0,
                       initial_mu1,
                       initial_sigma1,
                       initial_vertical_offset)
    
    # initialize curve fitting bounds
    amplitude_bounds = (noise_avg,np.max(y)-noise_avg*.25) 
    mu_bounds = (0,np.max(t)) ### maybe make these somewhat closer to the seeds
    sigma_bounds = (np.max(t)/150,np.max(t)/2) 
    vertical_offset_bounds = (.95*np.min( [np.min(y[:len(y)//5]), np.min(y[4*len(y)//5:])]), noise_avg+1.25*np.max( [np.max(y[:len(y)//5]), np.max(y[4*len(y)//5:])]) )
    bounds = bound_maker_erfrecterf(amplitude_bounds,mu_bounds,sigma_bounds,vertical_offset_bounds,number_erfrecterfs)
    initial_number_erfrecterfs = len(initial_sigma0)
    
    # make weights for fit
    sigma = make_weights(y, g_length=50)

    
    # limit the max number of function evaluations
    max_nfev = int(30*len(t))
    
    # try first fit
    try: 
        popt,pcov = curve_fit(lambda t,*p0:fit_wrapper_erfrecterf(t,*p0),t,y,p0=p0,bounds=bounds,x_scale=np.max(t),sigma=sigma,max_nfev=max_nfev,absolute_sigma=False)
    except Exception as e:
        print('p0:', p0)
        print('bounds', bounds)
        print('problem in first fit:', e)
        return
    ##### function will only reach this location if initial fit converged.
    
    # calculate residual 
    y_fit = fit_wrapper_erfrecterf(t,*popt)
    residual = y-y_fit
    

    
    """plt.figure()
    plt.plot(t, y, label="data")
    plt.plot(t, y_fit, label="1st fit")
    plt.plot(t, np.abs(residual)/sigma**2, label="|residual|/sigma**2")
    #plt.plot([0, np.max(t)], [noise_thresh, noise_thresh], 'k--', label="threshold")
    plt.plot([0, np.max(t)], [750, 750], 'k--', label="threshold")
    #plt.plot([0, np.max(t)], [noise_avg, noise_avg], 'k--', label="mean noise")

    plt.legend()"""
    
    """
    print(noise_thresh)
    print(np.any(np.abs(residual)>noise_thresh))
    print(number_gaussians<max_num_gaussians)
    """
    
    
    # compare residual to noise threshold to determine whether or not 
    # another Gaussian should be added. Only add another Gaussian if 
    # there are no more than max_num_gaussians Gaussians already. 
    std_dev_residual_previous = 9999999#noise_thresh-noise_avg#np.std(y)
    std_dev_residual_new = np.std(residual)

    fitnum = 1
    noisethresh_to_use = .05
    while (np.any(np.abs(residual)/sigma**2>noisethresh_to_use)) & (number_erfrecterfs<max_num_erfrecterfs) & (std_dev_residual_new<std_dev_residual_previous*.8):
        plt.figure()
        plt.plot(y, label='y')
        plt.plot(y_fit, label='fitted')
        plt.plot((y-y_fit)/sigma**2, label='scaled residual')
        plt.plot([0,len(y_fit)], [noisethresh_to_use, noisethresh_to_use], label='threshold')
        plt.legend()
        print('initial fit insufficient')
        # try subsequent fit
        # add in another gausian 
        fitnum += 1
        print('fit number', fitnum)
        try:
            new_a, new_mu0, new_sigma0, new_mu1, new_sigma1 = subsequent_seeding_erfrecterf(t, y, y_fit, popt, number_erfrecterfs, noise_thresh)


            initial_vertical_offset = popt[-1]

            p0 = initial_guess(new_a,
                               new_mu0,
                               new_sigma0,
                               new_mu1,
                               new_sigma1,
                               initial_vertical_offset)
            sigma0_bounds = (np.max(t)/150,np.max(t)/2) 
            sigma1_bounds = (np.max(t)/150,np.max(t)/2) 
            # initialize curve fitting bounds
            num_new_erfrecterfs = 1
            bounds = bound_maker_subsequent_erfrecterf(t, y, y_fit, popt, num_new_erfrecterfs, amplitude_bounds, 
                                            sigma0_bounds, sigma1_bounds, vertical_offset_bounds, new_mu0, new_mu1)

            # try curve fit again
        except Exception as e:
            print(e)
            print("$$$$$$$$$$$$$")
            return
        try:
            popt,pcov = curve_fit(lambda t,*p0:fit_wrapper_erfrecterf(t,*p0),t,y,p0=p0,bounds=bounds,x_scale=np.max(t),sigma=sigma,max_nfev=max_nfev,absolute_sigma=False)
        except: # if first fit fails to converge, end fitting
            print(selxn, "one of the subsequent fits failed to converge")
            return 
        y_fit = fit_wrapper_erfrecterf(t,*popt)
        residual = y-y_fit
        number_erfrecterfs += 1
        
       
        print('num erfs',number_erfrecterfs)
        
        std_dev_residual_previous = std_dev_residual_new
        std_dev_residual_new = np.std(residual)
        #print('std dev of residual is: ', std_dev_residual_new)
    
    if (np.any(np.abs(residual/sigma**2)<noisethresh_to_use) & (number_erfrecterfs<=max_num_erfrecterfs)):
        print(selxn, "WORKED")

        # package data for ML input.
        r2, rmse, max_error, max_error_normalized, duration = eval_fit(y, y_fit, t, popt, delta_t)
        data = package_fit_data_erfrecterf(r2, rmse, max_error, max_error_normalized, 
                                time_eff, duration, popt, fr, 
                                number_erfrecterfs, y, 
                                noise_avg, max_num_erfrecterfs, initial_number_erfrecterfs)

        """ plt.figure()
        plt.plot(t, y, label='signal')
        plt.plot(t, y_fit, label="fit")
        #plt.plot(t, np.abs(residual), label="|new residual|")
        plt.legend()"""
        #print(number_erfrecterfs)
        return data
    else:
        print(selxn, "max number of erfrecterfs reached, but fit not within noise threshold")
        return
    
    return
