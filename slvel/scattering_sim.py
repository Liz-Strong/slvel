import numpy as np
"""Simulate the light scattered by particles passing through simulated intensity fields.
"""

def radian_to_cartesian(r, theta):
    """
    Convert coordinates from radian to cartesian
    
    Args:
        r (float): radius
        theta (float): angle
    
    Returns:
        x (float): x value of coordinate
        y (float): y value of coordinate

    """

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    return x, y

def cartesian_to_radian(x, y):
    """
    Convert coordinates from cartesian to radian
    
    Args:
        x (float): x value of coordinate
        y (float): y value of coordinate
        
    
    Returns:
        r (float): radius
        theta (float): angle
    """
    
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y, x)
    
    return r, theta

def generate_rect_mask(xval,yval):
    """
    Make a mask to save 
    
    Args:
        xval (int): length of simulated intensity field grid [pixels]
        yval (int): length  of simulated intensity field grid [pixels]
    
    
    Returns:
        a mask the same size as the simulated intensity field
    """
    return np.ones((yval,xval), dtype=bool)


def roll_function(positions, I, angular_velocity):
    """
    Due to how the simulations are generated where the first point of the simulation 
    is at the smallest x value and the subsequent positions are in a clockwise 
    (counterclockwise) direction when the vorticity is positive (negative), the first
    point of the simulated intensity might lie in the middle of an intensity trace.  
    
    This needs to be compensated for by rolling array elements.  Simulations come onto 
    the screen from one of 4 sides.  Which side the sim comes onto the screen and 
    which side the sim leaves the screen defines how to roll the intensity as a function 
    of time such that the first returned position is at the entrance and the final returned 
    position is at the exit.  
    
    Args:
        positions (array): position  of Particle
        I (array): intensities calculated as a function of position
        angular velocity (float): Particle angular velocity
    
    Returns:
        p (array): position  of Particle, adjusted to preserve order of peaks
        I (array): intensities calculated as a function of p, adjusted to preserve order of peaks
    """
    p = positions.T
    x_0 = p[0][0]
    y_0 = p[1][0]
    
    clockwise = True
    if angular_velocity < 0:
        clockwise = False
    
    roll = 0
    if clockwise:
        if (x_0>0) and (y_0>0) and (y_0<616):
            # need to roll
            if 616/2 > y_0: # orbit starts in upper half of screen
                try:
                    rollval = -np.argwhere(p[1][:(len(p[1])//4+1)]==0)[0]
                    
                except IndexError: #if none of the points is actually equal to 0
                    rollval = -np.abs(p[1][:(len(p[1])//4+1)]).argmin()
                    
                p = np.roll(p,rollval,axis=1)
                I = np.roll(I,rollval)

            else: #orbit starts in middle or lower half of screen
                try:
                    rollval = np.argwhere(p[1]==616)[0]+len(p[1])//2
                except IndexError: #if none of the points is actually equal to 0
                    rollval = np.abs(p[1][3*(len(p[1])//4):]).argmin() 
                p = np.roll(p,rollval,axis=1)
                I = np.roll(I,rollval)

    else:
        print('need to implement this still... rolling for counterclockwise vorticity.')
        raise ValueError

    return p.T, I

def select_offsets(d_min, d_max, num_d, l, num_theta):
    """
    Set the center of the orbit relative to the center of the light
    sampling on the (r, theta) coordinate system, but returning coordinates
    in the (x, y) coordinate system
    
    Note that the angles are chosen such that they extend from -pi/(2l) to pi/(2l).
    This is to take advantage of the symmetry of the problem.
    
    Args:
        d_min (float): lower limit of D [pixels]
        d_max (float): upper limit of D [pixels]
        num_d (int): number of D to calculate
        l (int): oam azimuthal mode number
        num_theta (int): number of angles to calculate
    Returns:
        dx (array): array of x coordinates of D [pixels]
        dy (array): array of y coordinates of D [pixels]
        d (array): array of D, distance between light center and orbit center [pixels]
        theta (array): array of angles between light center and orbit center [pixels]
    
    """
    dx, dy, d, theta = [], [], [], []
    distances = np.sqrt(np.linspace(d_min**2, d_max**2, num_d)) # obey rdr for even sampling
    ###distances = (distances[1:]-distances[:-1])/2+distances[:-1]
    #angles = np.linspace(-np.pi/(2*l),  np.pi/(2*l), num_theta, endpoint=False) # sample evenly along theta
    angles = np.linspace(-np.pi/(2*l),  0, num_theta) # sample evenly along theta
    ###angles = (angles[1]-angles[0])/2 + angles[:-1]
    for dist in distances:
        for ang in angles:
            del_x, del_y = radian_to_cartesian(dist, ang)
            dx.append(del_x)
            dy.append(del_y)
            d.append(dist)
            theta.append(ang)
    return np.asarray(dx), np.asarray(dy), np.asarray(d), np.asarray(theta)

def select_orbit_radius(distance, angle, rp, r, num_radii=5):
    """
    Calculates num_radii orbit radii within permissible values given the size of the light and the sampling particle's center of orbit
    Ensures that the combinations of R and D generate particle trajectories which pass through the interference fringes
    
    Args:
        distance (float): D, distance between center of light and orbit center [pixels]
        angle (float): phi, angle between center of light and orbit center [rad]
        rp (float): approximate radius of interference fringes pattern, proportional to w_0*sqrt(np.abs(1+l)/2) [pixels]
        r (float): Particle radius [pixels]
        num_radii (int): number of radii to calculate between the smallest and largest permissible values
    
    Returns:
        orbit_radii (array): list of orbi radii R which are evenly spaced between the smallest and largest possible values
    """
    orbit_radii = {}
    #orbit_radii = [None]*len(distance)
    for i, (d, theta) in enumerate(zip(distance, angle)):
        orbit_r_min = d-rp-r
        orbit_r_max = d+rp+r
        
        orbit_radii[d] = np.linspace(orbit_r_min, orbit_r_max, num_radii)
    
    return orbit_radii

def select_offsets(d_min, d_max, num_d, l, num_theta):
    """
    Set the center of the orbit relative to the center of the light
    sampling on the (r, theta) coordinate system, but returning coordinates
    in the (x, y) coordinate system
    
    Note that the angles are chosen such that they extend from -pi/(2l) to pi/(2l).
    This is to take advantage of the symmetry of the problem.
    
    Args:
        d_min (float): smallest possible D [pixels]
        d_max (float): largest possible D [pixels]
        num_d (int): number of D to calculate
        l (int): OAM azimuthal mode number
        num_theta (int): number of angles to calculate
    
    Returns:
        dx (array): x coordinate of D, the distance between the center of light and the center of orbit [pixels]
        dy (array): y coordinate of D, the distance between the center of light and the center of orbit [pixels]
        d (array): D, the distance between the center of light and the center of orbit [pixels]
        theta (array): angle between the center of light and the center of orbit [rad]
    """
    dx, dy, d, theta = [], [], [], []
    distances = np.sqrt(np.linspace(d_min**2, d_max**2, num_d)) # obey rdr for even sampling
    ###distances = (distances[1:]-distances[:-1])/2+distances[:-1]
    #angles = np.linspace(-np.pi/(2*l),  np.pi/(2*l), num_theta, endpoint=False) # sample evenly along theta
    angles = np.linspace(-np.pi/(2*l),  0, num_theta) # sample evenly along theta
    ###angles = (angles[1]-angles[0])/2 + angles[:-1]
    for dist in distances:
        for ang in angles:
            del_x, del_y = radian_to_cartesian(dist, ang)
            dx.append(del_x)
            dy.append(del_y)
            d.append(dist)
            theta.append(ang)
    return np.asarray(dx), np.asarray(dy), np.asarray(d), np.asarray(theta)


def select_random_on_anulus(lower, upper, number=1):
    """
    Select randomly on an annulus without bias 
    
    Args:
        lower (float): smaller annulus radius
        upper (float): larger annulus radius
        number (int): number of samples to take
    
    Returns:
        random_number (array): number selected at random on annulus without bias
    """
    random_number = [np.sqrt(random.uniform(lower**2, upper**2)) for i in range(number)]
    if number==1:
        random_number=random_number[0]
    return random_number

def select_random_R(D, l, w0, R_min, R_max, c=1.35):
    """
    Select an orbit radius at random from an evenly sampled annular spaced between R_min and R_max 
    
    Args:
        D (float): Distance between orbit center and center of light
        l (int) OAM azimuthal mode number
        w0 (float): beam waist
        R_min (float): lower limit of R
        R_max (float): upper limit of R
    
    
    Returns:
        R (float): orbit radius R selected at random on annulus
    """
    R = []
    r1 = D-c*w0*np.sqrt((l+1)/2)
    r2 = D+c*w0*np.sqrt((l+1)/2)

    r1[r1<R_min]=R_min
    r2[r2<R_min]=R_min
    r2[r2>R_max]=R_max
    r1[r1>R_max]=R_max
    
    for r_min, r_max in zip(r1, r2):
        r_selected = select_random_on_anulus(r_min, r_max)#random.uniform(r_min, r_max)
        R.append(r_selected) 
    return R

def save_series_info(info, intensities_keep, intensities_new):
    """
    make a time series that has the info (eg radius or offset) for each burst
    
    Args:
        info (array): array to package to save 
        intensities_keep (array):
        intensities_new (array): 
    
    
    Returns:
        saving_info (array): info, packaged for saving for burst detection
    """
    shapes = [np.shape(intensities_new[i])[0] for i in range(np.shape(intensities_keep)[0]) ]
    saving_info = []
    for i, v in enumerate(info):
        saving_info.append(v*np.ones(shapes[i]))
    saving_info = np.concatenate(saving_info)
    saving_info = np.append(np.zeros(1000), saving_info)
    saving_info = np.insert(saving_info, 0, np.zeros(500))
    return saving_info

def simulate_time_series(p1, psi, intensity_shape, offset_x, offset_y, D, angle, angular_velocities, orbit_radius, xval, yval):
    """
    Simulates a burst given simulation parameters
    
    Args:
        p1 (Particle class): sampling particle
        psi (array): presummed intensity [intensity units]
        intensity_shape (array): shape of intensity field [pixels]
        offset_x (array): array of x coordinates of D [pixels]
        offset_y (array): array of y coordinates of D [pixels]
        D (array):  array of distances between center of light & center of orbit  [pixels]
        angle (array): array of angles between center of light & center of orbit [rad/sec]
        angular_velocities (array): array of angular velocities [rad/sec]
        orbit_radius (array): array of orbit radii [pixels]
        xval (int): x dimension of grid on which intensity is simulated [pixels]
        yval (int): y dimension of grid on which intensity is simulated [pixels]
        
    Returns:
        
        time_keep (array): times corresponding to each simulated burst [sec]
        intensities_keep (array): intensities of each simulated burst [intensity units]
        positions_keep (array): position of particle at each time for each simulated burst [pixels]
        angular_velocities_keep (array): angular velocities of each simulated burst [rad/sec]
        R_keep (array): orbit radii of each simulated burst [pixels]
        d_keep (array): distance D between center of light & center of orbit [pixels]
        offset_x_keep (array): x coordinates of D of each simulated burst [pixels]
        offset_y_keep (array): y coordinates of D of each simulated burst [pixels]
        theta_keep (array): angle between center of light & center of orbit [rad]
    """
    # make arrays in which to store data in loop below
    intensities_keep = []
    time_keep = []
    positions_keep = []
    angular_velocities_keep = []
    R_keep = []
    d_keep = []
    theta_keep = []
    offset_x_keep = []
    offset_y_keep = []

    # make mask for use below
    mask = generate_rect_mask(xval, yval)

    # iterate through the parameters, calculating a burst for each 
    for i, (delta_x, delta_y, offset_d, theta, angvel, R) in enumerate(zip(offset_x, 
                                                                    offset_y, 
                                                                    D, 
                                                                    angle, 
                                                                    angular_velocities,
                                                                    orbit_radius)):
        # set attributes of particle
        p1.orbit_offset_x, p1.orbit_offset_y = delta_x, delta_y
        p1.orbit_radius = R
        p1.v = angvel

        # calculate intensity & positions as functions of time
        I,positions,time = p1.sample_on_orbit_ps(psi, intensity_shape)

        # roll values to maintain expected order of peaks.. necessary because simulation 
        # starts on leftmost point
        positions, I_rolled = roll_function(positions, I, angvel)

        ## save only values within the mask: eliminate the tails that have signals that are close to 0
        # need to eliminate all positions[:,1]>len(mask[0,:]) & positions[:,0]>len(mask[:,0])
        # also only save the values that are on the image.
        g = (positions.T[0]>0) & (positions.T[0]<xval)
        h = (positions.T[1]>0) & (positions.T[1]<yval)
        positions = positions[g&h]     # note we do NOT want to do this if we're not using the mask
        intensities = np.array(I_rolled)
        intensities = intensities[g&h] # note we do NOT want to do this if we're not using the mask

        values_to_save = mask[positions[:,1],positions[:,0]];


        intensities_maskd = intensities[values_to_save]

        # save values when the signals are somewhat long enough
        if len(intensities[values_to_save]) > 25:
            intensities_keep.append(intensities[values_to_save])
            positions_keep.append(positions[values_to_save])

            time_keep.append(time[np.arange(0,len(positions[values_to_save]))])
            angular_velocities_keep.append(p1.v)
            R_keep.append(R)
            d_keep.append(offset_d)
            offset_x_keep.append(delta_x)
            offset_y_keep.append(delta_y)
            theta_keep.append(theta)

            #plt.figure(1),plt.plot(positions[:,0],positions[:,1],'x')
            #plt.figure(2),plt.plot(time,I)
        else:
            print("too short")
    return time_keep, intensities_keep, positions_keep, angular_velocities_keep, R_keep, d_keep, offset_x_keep, offset_y_keep, theta_keep 

def concat_timeseries(intensities_keep, time_keep, ext_length=420):
    """
    Concatenates the calculated intensities into a time series
    
    Args:
        intensities_keep (array): the simulated timeseries
        time_keep (array): times corresponding to intensities_keep
        ext_length (int): the spacing between bursts is based on this number. Should be at least 1.5x as long as the longest expected burst.
        
    Returns:
        timeseries_time (array): times to accompany timeseries_intensity
        timeseries_intensity (array): bursts concattenated into a single time series
        intensities_extended (array): the extended bursts. used later when tagging the portions of the time series with the known simulation parameters.
    """
    
    # extend simulated bursts to space them appropriately
    intensities_extended = [np.append(intensities_keep[i],np.zeros(np.random.randint(ext_length,high=ext_length*1.05))) for i in range(np.shape(intensities_keep)[0])]
    
    # concatenate extended bursts
    tmp = np.concatenate(intensities_extended)
    
    # add some zeros at the begining to use these to estimate noise levels later
    tmp = np.append(np.zeros(1000), tmp)
    tmp = np.insert(tmp, 0, np.zeros(500))
    
    # add noise
    noise = np.random.normal(0,2,len(tmp))
    tmp += noise

    # calculate time for entire time series
    timeseries_time = (time_keep[0][1]-time_keep[0][0]) * np.arange(len(tmp))
    
    # timeseries is all positive 
    timeseries_intensity = tmp - np.min(tmp) 
    return timeseries_time, timeseries_intensity, intensities_extended
