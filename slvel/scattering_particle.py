import numpy as np
"""A particle blurs the intensity field. 
"""    
class Particle():
    """
    Attributes:
        intensity_shape (array): shape of intensity of scene to be sampled by Particle. each element represents 1 pixel
        particle_radius (float): radius of particle [pixels] 
        orbit_radius (float): radius of orbit [pixels]
        orbit_offset_x (float): x position of orbit center referenced from light center [pixels]
        orbit_offset_y (float): y position of orbit center referenced from light center [pixels]
        orbit_centerx (float): x position of orbit referenced from center of intensity_shape [pixels]
        orbit_centery (float): y position of orbit referenced from center of intensity_shape [pixels]
        v (float): angular velocity of particle [rad/sec]
        sample_rate (float): how frequently samples are collected [Hz]
        direction (int): either 1 or -1
    """
    
    
    def __init__(self,intensity_shape,particle_radius=5,orbit_radius=50,orbit_offset_x=0,orbit_offset_y=0,v=50,sample_rate=100000,direction=1):
        
        # each particle has some size 
        self.particle_radius = particle_radius
        # each particle is located on an orbit with radius & offset from light center
        self.orbit_radius = orbit_radius
        self.orbit_offset_x = orbit_offset_x
        self.orbit_offset_y = orbit_offset_y
        # each particle has some angular velocity v
        self.v = v
        # each particle is sampled at a specific rate
        self.sample_rate = sample_rate #Hz
        # each particle travels in a specific direction (+1 or -1)
        self.direction = direction
              
    def reset_position(self,intensity_shape):
        """
        Place the particle at the starting position. Set orbit_centerx and orbit_centery
        
        Args:
            intensity_shape (array): intensity of scene to be sampled by Particle. each element represents 1 pixel

        """
        # each particle is initially located on the orbit
        self.x = int(np.round(self.orbit_offset_x + self.orbit_radius))
        self.y = int(np.round(self.orbit_offset_y))
        # each orbit has a center 
        self.orbit_centerx = intensity_shape[1]/2. + self.orbit_offset_x
        self.orbit_centery = intensity_shape[0]/2. + self.orbit_offset_y
        
    def calculate_positions(self,Nsamples):
        """
        Given the number of samples to take along an orbit, calculate the positions of these sampels
        
        Args:
            Nsamples (int): number of samples to be taken along orbit
        
        Returns:
            positions (array): list of positions where the center of the particle should be for sampling
        """
        # segment orbit into Nsamples evenly angularly spaced
        centerX = [int(np.round(-self.orbit_radius*np.cos(2*np.pi/Nsamples*fn) + self.orbit_centerx)) for fn in range(Nsamples)]
        centerY = [int(np.round(self.orbit_radius*np.sin(2*np.pi/Nsamples*fn)*self.direction + self.orbit_centery)) for fn in range(Nsamples)]
        positions = np.array([centerX,centerY]).T
        return positions
        
    def sample_on_orbit(self,intensity_normalized,intensity_shape):
        """
        Calculates the intensity sampled by the particle as a function of time
        
        Args:
            intensity_normalized (array): intensity scene to be sampled by Particle
            intensity_shape (array): shape of intensity of scene to be sampled by Particle. each element represents 1 pixel
        
        Returns:
            I (array): list of measured intensities at each position
            positions (array): list of particle sampling positions
            time (array): list of time stamps for each sample
        """
        #make sure the particle is at the starting position. calculate the orbit centers. 
        self.reset_position(intensity_shape)
        # number of samples to collect over orbit
        Ns = int(np.round(self.sample_rate/self.v*2*np.pi))
        positions = self.calculate_positions(Nsamples=Ns)
        I=[]
        for i in range(len(positions)):
            # move center to position
            self.x,self.y = positions[i][0],positions[i][1]
            I.append(self.calculate_intensity(intensity_normalized))

        time = 1/self.sample_rate * np.linspace(0,Ns,Ns)
        return I,positions,time
        
        
    def find_points_in_sample(self):
        """
        Determines which points are located within the Particle
        
        Returns:
            xvals (list): ordered list of y coordinates of pixels located within Particle
            yvals (list): ordered list of y coordinates of pixels located within Particle
        """
        xvals = np.linspace((self.x-2*self.particle_radius)-1,(self.x+2*self.particle_radius+1),self.particle_radius*4+3)
        yvals = np.linspace((self.y-2*self.particle_radius)-1,(self.y+2*self.particle_radius+1),self.particle_radius*4+3)
        [xx,yy] = np.meshgrid(xvals,yvals)
        squaredvalues = (xx-self.x)**2+(yy-self.y)**2
        squaredvaluestokeep = (squaredvalues<self.particle_radius**2) 
        valuestokeep_indices = np.where(squaredvaluestokeep)
        
        xvals = np.array([*map(int,np.round(xx[valuestokeep_indices]))])
        yvals = np.array([*map(int,np.round(yy[valuestokeep_indices]))])
        
        return xvals,yvals
        
        
    def calculate_intensity(self,intensity_normalized):
        """
        Calculates the summed intensity of the intensity field sampled by the Particle
        
        Args:
            intensity_normalized (array): intensity field to be sampled
            
        Returns:
            I (float): total intensity sampled by Particle
        """
        
        xpoints,ypoints = self.find_points_in_sample()

        #[xuse for xuse in xpoints if (xuse <= 616 and xuse >= 0)]
        c1 = xpoints >= 0
        c2 = xpoints <= intensity_normalized.shape[1]-1
        c3 = ypoints >= 0
        c4 = ypoints <= intensity_normalized.shape[0]-1
        conds = np.array([c1.T,c2.T,c3.T,c4.T])
        pointsInGrid = np.all(conds,axis=0)
        
        #xpoints = xpoints[boolArr]
        #ypoints = ypoints[boolArr]
        xIn = xpoints[pointsInGrid]
        yIn = ypoints[pointsInGrid]
        intensities = intensity_normalized[yIn,xIn] #REVERSED 
        I = 0
        I = np.sum(intensities)

        return I
    

    def calculate_sampled_intensities_throughout(self,intensity_normalized):
        """
        Calculates the convolution of the sampling particle with the intensity array.
        This is a time consuming process, so this only needs to be done once. Save 
        its results in a file and then sampling these results along the trajectories
        of the particle is much more time efficient than summing all the intensities at
        every location along every orbit trajectory. Presumming the intensities eliminates 
        a lot of redundant calculations.
        
        Args:
            intensity_normalized (array): intensity field
        
        Returns:
            presummed_intensities (array): the intensity field blurred by the sampling particle.
                Each value corresponds to the summed intensity of intensity_normalized when
                the particle is at that position. 
        
        """
        #presummed_intensities = np.zeros(np.shape(intensity_normalized))
        presummed_intensities = np.zeros( (np.shape(intensity_normalized)[0]+2*self.particle_radius,
                                   np.shape(intensity_normalized)[1]+2*self.particle_radius) )
        # the coordinate conventions are weird.. figured it out by looking 
        # above at line 127.
        for x in range(-self.particle_radius, np.shape(intensity_normalized)[1]+self.particle_radius):
            for y in range(-self.particle_radius, np.shape(intensity_normalized)[0]+self.particle_radius):
                self.x, self.y = x,y
                presummed_intensities[y][x] = self.calculate_intensity(intensity_normalized)
        return presummed_intensities
    
    
    def select_intensity_ps(self,presummed_intensities,psi_shape):
        """
        Calculates the intensity at the Particle position given the convolution of the 
        Particle with the intensity
        
        Args:
            presummed_intensities (array): as calculated in calculate_sampled_intensities_throughout. The convolution of the partile with the intensity field
            psi_shape: shape of presummed_intensities
            
        Returns:
            intensity at the position of the Particle
        """
        if (self.y < psi_shape[0]) and (self.y >= 0) and (self.x < psi_shape[1]) and (self.x >= 0):
            return presummed_intensities[self.y][self.x]
        else:
            return 0
    
    def sample_on_orbit_ps(self,presummed_intensities,
                           intensity_shape):
        """
        Calculates the intensity sampled by the particle as a function of time. Similar to sample_on_orbit
        but is much faster as it uses the pre-calculated presummed_intensities and does not need to calculate
        redundant parts of this convolution. 
        
        Args:
            presummed_intensities (array): as calculated in calculate_sampled_intensities_throughout. 
                the convolution of the partile with the intensity field
                intensity_shape (array): the sahape of presummed_intensities
                
        Returns:
            I (array): list of measured intensities at each position
            positions (array): list of particle sampling positions
            time (array): list of time stamps for each sample

        """
        #make sure the particle is at the starting position. calculate the orbit centers. 
        self.reset_position(intensity_shape)
        # number of samples to collect over orbit
        Ns = int(np.round(self.sample_rate/self.v*2*np.pi))
        positions = self.calculate_positions(Nsamples=Ns) 
        I=[]
        presummed_intensities_shape = np.shape(presummed_intensities)
        for i in range(len(positions)):
            # move center to position
            self.x,self.y = positions[i][0],positions[i][1]
            I.append(self.select_intensity_ps(presummed_intensities,
                                              presummed_intensities_shape))

        time = 1/self.sample_rate * np.linspace(0,Ns,Ns)
        return I,positions,time
 