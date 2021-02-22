"""Identify burst locations within signal.

Given a signal, experimental or simulated, identify where bursts start and end.
"""
import numpy as np

def gaussian_generator(npoints,std):
    """
    Make a gaussian f npoints long with standard deviation std
    
    Args:
        npoints (int): length of Gaussian
        std (float): standard deviation of Gaussian
        
    Returns:
        a Gaussian function
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
        a rect function
    """
    r = np.zeros(npoints)
    r[int(np.floor(npoints/2-width/2+1)):int(np.ceil(npoints/2+width/2))] = area/width
    return r

def find_above_threshold(xc_r, threshold):
    """
    Finds whre the cross correlation is greater than the threshold. Like a zero crossing algorithm.
    
    Args:
        xc_r (array): cross correlation between signal and rect
        threshold (float): function notes when xc_r is greater than this value
        
    Returns:
        a list that's either 0 if xc_r<=threshold or 1 if xc_r>threshold
    """
    thresholded = [i for i,x in enumerate(xc_r) if x>threshold]
    where_above_threshold = np.zeros(len(xc_r))
    where_above_threshold[thresholded] = 1
    return where_above_threshold

def segment_analysis(above_threshold,intensity):
    """
    Determines the start, end, length, and area of each burst. 
    
    Args:
        above_threshold (array): array with values 0 if xc_r<=threshold or 1 if xc_r>threshold
        intensity (array): time series under study
    
    Returns:
        segment_start (array): locations of all the starts of the segments
        segment_end (array): locations of all the ends of the segments
        segment_length (array): segment_end-segment_start for each segment
        segment_area (array): summed intensity within each segment
        binary_segment_locations (array of bools): boolean of segment_locations
    """
    # take the derivative of the binary vecor above_threshold.  
    # This will indicate where transitions from zeros to ones occur.
    # Bursts start where the derivative is equal to .5
    # Bursts end where the derivative is equal to -.5
    gradient = np.gradient(above_threshold)
    segment_starts = [x for x,val in enumerate(gradient) if val==.5]
    # select every other since for each start there are two consecutive points of 0.5.. redundant info.
    segment_start = segment_starts[::2]
    
    segment_ends = [x for x,val in enumerate(gradient) if val==-.5]
    # select every other since for each start there are two consecutive points of 0.5.. redundant info.
    segment_end = segment_ends[::2]
    
    segment_length = [segment_end[i]-segment_start[i] for i,val in enumerate(segment_start)]
    
    segment_area = [np.sum(intensity[segment_start[j]:segment_end[j]]) for j in range(len(segment_start))]
    
    # make binary list 
    segment_locations = np.gradient(np.cumsum(above_threshold))
    binary_segment_locations = [bool(i) for i in segment_locations]
    return segment_start,segment_end,segment_length,segment_area,binary_segment_locations

def plot_segments(y,intensity,threshold):
    """
    Plots the elements used to analyzed the bursts
    
    Args:
        y (array): array of above_threshold which has values 0 if xc_r<=threshold or 1 if xc_r>threshold
        intensity (array): summed intensity
        threshold (float): value used to generate y. 
    Returns:
        None
    """
    gradient = np.gradient(y)
    plt.figure()
    plt.plot(y/56000,'r')
    plt.plot(gradient,'k')
    plt.plot(intensity/1000)
    plt.title('derivative')
    big_changes = np.where(gradient>threshold)
    return None

def eliminate_bad_segments(starts, ends, bad_start_locns):
    """
    Delete segments which are deemed bad
    
    Args:
        starts (array): list of segment start locations
        ends (aray): list of segment end locations
        bad_start_locns (array): list of start locations which correspond to bad segments
        
    Returns:
        starts (array): list of segment start locations, now with bad segments deleted
        ends (array): list of segment end locations, now with bad segments deleted
    """
    starts = np.delete(starts, bad_start_locns)
    ends = np.delete(ends, bad_start_locns)
    print("\n\n\n******\nSTARTS AND ENDS CLEANED UP\n******")
    return starts, ends

def inspect_bursts(xc_r, starts, ends, ang_velocities, min_dist_between_start_end=50):
    """
    An entire time series should only have as many starts as it has ang_velocities
    
    Args:
        xc_r (array): cross correlation between signal and rect
        starts (array): list of segment start locations
        ends (array): list of segment end locations
        ang_velocities (array): list of angular velocities of simulated bursts
        min_dist_between_start_end (float): minimum length of a burst
    
    Returns:
        bad_burst_start_locns (array): list of start locations of segments which don't adhere to criteria to deem them bursts
    """
    
    print('number of detected segments:', len(starts))
    print('number of simulated segments:', len(ang_velocities))
    if (len(starts) != len(ang_velocities)):
        print('The number of detected segments should equal the number of simulated segments. Change the parameters of the burst detection Gaussian correlator.')
    
    dist_between_segs = np.asarray(starts[1:])-np.asarray(ends[:-1])
    if np.min(dist_between_segs)<min_dist_between_start_end:
        print('segments are too close to each other: they are within the buffer of', min_dist_between_start_end, 'data pionts of each other')
        
        bad_burst_start_locns_half = np.ndarray.flatten(np.argwhere(dist_between_segs<min_dist_between_start_end))
        bad_burst_start_locns = np.append(bad_burst_start_locns_half, bad_burst_start_locns_half + 1) # here we'll say that both the segments that abut each other are bad. 
        bad_burst_starts = np.array([starts[i] for i in bad_burst_start_locns])
        print('bad segments start at:', np.sort(bad_burst_starts))
        
        #bad_burst_area = np.array([areas[i] for i in bad_burst_start_locns])
        #print('areas of bad bursts: ', bad_burst_areas)
        print('bad segment numbers: ', np.sort(np.append(bad_burst_start_locns, bad_burst_start_locns+1)))
    return bad_burst_start_locns

def segment_maker(intensity,starts,ends,buffer,min_length=25):
    """
    given the start and end of each burst's rough position, add the number of points specified in the buffer to create a segment that includes tails on both sides of the burst. The segment should have a substantial amount of tail for the autocorrelation to work well.

    Args:
        intensity (array): signal of interest
        starts (array): position in intensity array of start of each rough location
        ends (array): position in intensity array of end of each rough location
        buffer (float): number of points on either side of rough location to be included in the segment
        min_length (float): minimum burst length
    
    Returns:
    	segs (array): list of segments. each element in list is a list of intensities in a segment.
    """
    # when the buffer added to the start of the first and to the end of the last 
    # rough location are located in the intensity array
    if (starts[0]-buffer)>=0 & (ends[-1]+buffer)<=len(intensity):
        # each segment is the rough location with the buffer on either side.
        segments = [intensity[s-buffer:e+buffer] for s,e in zip(starts,ends)]
        
    # when the buffer added to the start of the first is out of the intensity array 
    # but the buffer added to the end of the last rough location is in the intensity array
    elif (starts[0]-buffer)<0 & (ends[-1]+buffer)<=len(intensity):
        seg1 = intensity[starts[0]:ends[0]+buffer] 
        seg2 = [intensity[s-buffer:e+buffer] for s,e in zip(starts[1:],ends[1:])]
        segments = [seg1] + seg2
        
    # when the buffer added to the start of the first is in the intensity array
    # but the buffer added to teh end of the last rough location is out of the intensity array
    elif (starts[0]-buffer)>=0 & (ends[-1]+buffer)>len(intensity):
        seg1 = [intensity[s-buffer:e+buffer] for s,e in zip(starts[:-1],ends[:-1])]
        seg2 = intensity[starts[-1]-buffer:ends[-1]]
        segments = seg1 + [seg2]
        
    # when neither the buffer added to the first or the last of the start and end, respectively, 
    # of the rough locations are in the intensity array
    else:
        seg1 = intensity[starts[0]:ends[0]+buffer] 
        seg2 = [intensity[s-buffer:e+buffer] for s,e in zip(starts[1:-1],ends[1:-1])]
        seg3 = intensity[starts[-1]-buffer:ends[-1]]
        segments = [seg1] + seg2 + [seg3]
        
    # save only segments that are long enough
    seg_min_length = 2*buffer + min_length
    segment_length_criteria = np.array([np.shape(segments[i])[0]>seg_min_length for i in range(len(segments))])
    segs = np.asarray(segments)[segment_length_criteria]
    if len(segs) == 0:
        return
    return segs