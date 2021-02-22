# slvel

*structured light velocimetry*

*slvel* 
When particles pass through beams of light, the light they scatter can be studied to deduce their kinematic histories. This principle forms the backbone of flow velocimetry techniques like laser Doppler velocimetry (LDV). Here, we extend this concept to accommodate complicated illumination patterns and expected particle movements. In particular, we simulate particle trajectories through structured light. We then identify and parameterize these simulations. A machine learning regression algorithm trained on simulated data predicts the angular velocities of the scattering partiles.

More documentation is [here](https://slvel.readthedocs.io/)

### Package details:
- `slvel/` contains the Python modules for `slvel`. 
- `docs/` contains the code for generating the project documentation.
- `examples/` contains Jupyter notebooks which set up simulations for studying the kinematics of reflective particles as they pass through beams of structured light.

*slvel* requires Python 3.7, [`numpy`](https://numpy.org/), [`scipy`](https://scipy.org/), [`tensorflow`](https://tensorflow.org/), [`matplotlib`](https://matplotlib.org/), [`pandas`](https://pandas.pydata.org/), and [`seaborn`](https://seaborn.pydata.org/)

### Authors:

*slvel* was written by Elizabeth Strong. 

This code accompanies a paper that is curretnly in press called *Angular velocimetry for fluid flows: an optical sensor using structured light and machine learning* by Elizabeth F. Strong, Alex Q. Anderson, Michael P. Brenner, Brendan M. Heffernan, Nazanin Hoghooghi, Juliet T. Gopinath, and Greg B. Rieker.

### How to Cite:

Strong, E.F **slvel* **2021**, [www.github.com/Liz-Strong/slvel] (www.github.com/Liz-Strong/slvel).

### License:

This project is licensed under the BSD 3-Clause License. Please see `LICENSE` for full terms and conditions. 
