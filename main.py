# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point
import parameters as pm
from solver import BESSER
from tqdm import tqdm
import constants as c

# Import needed libraries
import numpy as np
import os

# set numpy to raise all as errors
# np.seterr(all='raise')

def main(pm=pm):
    """
    Main code
    """

    # Initializating state and RT coefficients
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)
    st = state(cdt)

    # Create path for output files
    if not os.path.exists(pm.dir):
        os.makedirs(pm.dir)
    datadir = pm.dir + 'out'
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    datadir = pm.dir + 'out/'

    # Clean
    for fil in os.listdir(datadir):
        if not os.path.isdir(datadir+fil):
            os.remove(datadir+fil)

    # Save the parameters used in this run to a file
    # save just the parameters module to the output directory
    # first convert the parameters to a dictionary
    module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
    pm_dict = module_to_dict(pm)
    # save the dictionary to a file
    f = open(datadir+'parameters.out', 'w')
    for key, value in pm_dict.items():
        if type(value) != dict:
            f.write(f'{key:<25}: {value}\n')
    f.close()

    # Opening MRC file
    f = open(datadir+'MRC', 'w')
    f.write(f'Itteration  ||  Max. Rel. change\n')
    f.write('-'*50 + '\n')
    f.close()

    # Compute the initial conditions and save them to file to check
    for KK in range(3):
        for QQ in range(KK+1):
            # Write the JKQ of each height into a file
            prof_real = np.zeros((1 + cdt.z_N,len(st.radiation[0].nus)))
            prof_real[0,:] = 1e7*c.c/st.radiation[0].nus
            prof_imag = np.zeros((1 + cdt.z_N,len(st.radiation[0].nus)))
            prof_imag[0,:] = 1e7*c.c/st.radiation[0].nus

    # Start the main loop for the Lambda iteration
    for itteration in range(cdt.max_iter):

        print('-'*80)
        print(f'New itteration {itteration}')
        print('-'*80)
        print(f'Reseting radiation')
        # Reset the internal state for a new itteration
        st.new_itter()

        # Go through all the rays in the cuadrature
        for j, ray in tqdm(enumerate(cdt.rays), ncols=80, total=len(cdt.rays), desc=f'Propaating rays'):

            # Initialize optical depth
            tau = np.zeros((cdt.nus_N))

            # Reset lineal
            lineal = False
            tau_tot = [0.]

            # Define limits in height index and direction
            if ray.is_downward:
                step = -1
                iz0 = -1
                iz1 = -cdt.z_N - 1
            else:
                step = 1
                iz0 = 0
                iz1 = cdt.z_N

            # Deal with very first point computing first and second
            z = iz0
            # Allocate point
            point_O = point(st.atomic[z], st.radiation[z], cdt.zz[z])

            # If top boundary
            if iz0 == -1:
                # Set Stokes
                point_O.setradiationas(st.space_rad)

            # If bottom boundary
            elif iz0 == 0:
                # Set Stokes
                point_O.setradiationas(st.sun_rad[j])
                point_O.sumStokes(ray,cdt.nus_weights,cdt.JS)

            # Get RT coefficients at initial point
            sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)

            # Get next point and its RT coefficients
            z += step
            point_P = point(st.atomic[z], st.radiation[z], cdt.zz[z])
            sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

            # Go through all the points (besides 0 and -1 for being IC)
            for z in range(iz0+step, iz1, step):

                # Shift data
                point_M = point_O
                point_O = point_P
                sf_m = sf_o
                kk_m = kk_o
                sf_o = sf_p
                kk_o = kk_p
                kk_p = None
                sf_p = None

                if z == iz1 - step:
                    point_P = False
                    lineal = True

                else:
                    point_P = point(st.atomic[z+step], st.radiation[z+step], \
                                    cdt.zz[z+step])

                    # Compute the RT coeficients for the next point (for solving RTE)
                    sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

                # Propagate
                tau_tot = BESSER(point_M, point_O, point_P, \
                                 sf_m, sf_o, sf_p, \
                                 kk_m, kk_o, kk_p, \
                                 ray, cdt, tau_tot, not lineal, \
                                 tau)

                # Add to Jqq
                point_O.sumStokes(ray,cdt.nus_weights,cdt.JS)

        print(f'Finished propagating rays')
        print(f'Solving ESE at each point')
        # Update the MRC and check wether we reached convergence
        st.update_mrc(cdt, itteration)

        print(f'Finished solving ESE')
        print(f'Saving MRC')
        # Write into file
        f = open(datadir+f'MRC','a')
        f.write(f'{itteration:4d}   {st.mrc_p:14.8e}  {st.mrc_c:14.8e}\n')
        f.close()

        # If converged
        if (st.mrc_p < cdt.tolerance_p and st.mrc_c < cdt.tolerance_c):
            print('\n----------------------------------')
            print(f'FINISHED WITH A TOLERANCE OF {st.mrc_p};{st.mrc_c}')
            print('----------------------------------')
            break
        elif itteration == cdt.max_iter-1:
            print('\n----------------------------------')
            print(f'FINISHED DUE TO MAXIMUM ITERATIONS {cdt.max_iter}')
            print('----------------------------------')

    # Remove unused boundaries
    for j in cdt.rays:
        st.sun_rad.pop(0)

    print(f'Finished Lambda iteration')
    print(f'Computing emergent radiation')
    # Go through all the rays in the emergent directions
    outputs = []
    # for j, ray in enumerate(tqdm(cdt.orays, desc='emerging rays', leave=False)):
    for j, ray in tqdm(enumerate(cdt.orays), ncols=80, total=len(cdt.orays), desc=f'Computing emergent radiation'):

        # Initialize optical depth
        tau = np.zeros((cdt.nus_N))

        # Reset lineal
        lineal = False

        # Define limits in height index and direction
        if ray.is_downward:
            step = -1
            iz0 = -1
            iz1 = -cdt.z_N - 1
        else:
            step = 1
            iz0 = 0
            iz1 = cdt.z_N


        # Deal with very first point computing first and second
        point_O = point(st.atomic[z], st.radiation[z], cdt.zz[z])
        z = iz0
        if iz0 == -1:
            point_O.setradiationas(st.space_rad)
        elif iz0 == 0:
            point_O.setradiationas(st.osun_rad[j])

        # Get RT coefficients at initial point
        sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)

        # Get next point and its RT coefficients
        z += step
        point_P = point(st.atomic[z], st.radiation[z], cdt.zz[z])
        sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

        # Go through all the points (besides 0 and -1 for being IC)
        for z in range(iz0, iz1, step):

            # Shift data
            point_M = point_O
            point_O = point_P
            sf_m = sf_o
            kk_m = kk_o
            sf_o = sf_p
            kk_o = kk_p
            kk_p = None
            sf_p = None

            if z == iz1 - step:
                point_P = False
                lineal = True
            else:
                point_P = point(st.atomic[z+step], st.radiation[z+step], \
                                cdt.zz[z+step])
                # Compute the RT coeficients for the next point (for solving RTE)
                sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

            # Transfer
            tau_tot = BESSER(point_M, point_O, point_P, \
                             sf_m, sf_o, sf_p, \
                             kk_m, kk_o, kk_p, \
                             ray, cdt, tau_tot, not lineal, \
                             tau)

        # Store last Stokes parameters
        f = open(datadir + f'stokes_{j:02d}.out', 'w')
        N_nus = point_O.radiation.nus.size
        f.write(f'Number of frequencies:\t{N_nus}\n')
        f.write(f'frequencies(cgs)\t\tI\t\tQ\t\tU\t\tV\n')
        f.write('----------------------------------------------------------\n')
        for i in range(N_nus):
            f.write(f'{point_O.radiation.nus[i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[0][i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[1][i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[2][i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[3][i]:25.16e}\n')
        f.close()

        f = open(datadir + f'tau_{j:02d}.out', 'w')
        N_nus = point_O.radiation.nus.size
        f.write(f'Number of wavelengths:\t{N_nus}\n')
        f.write(f'wavelengths(nm)\ttau\n')
        f.write('----------------------------------------------------------\n')
        for nu,ta in zip(point_O.radiation.nus,tau):
            f.write(f'{1e7*c.c/nu:25.16f}  {ta:23.16e}\n')
        f.close()

        # add the ray, wavelegnths, taus, and stokes to a variable and output it
        outputs.append([ray, point_O.radiation.nus, cdt.zz, tau, point_O.radiation.stokes])
    
    print(f'Finished emergent radiation')
    print(f'ENJOY YOUR RESULTS :)')

    return outputs

if __name__ == '__main__':
    main()
