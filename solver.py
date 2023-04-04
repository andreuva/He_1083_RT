import copy
import numpy as np
import constants as c


def BESSER(point_M, point_O, point_P, \
           sf_m, sf_o, sf_p, \
           kk_m, kk_o, kk_p, \
           ray, cdt, tau_tot, quad, \
           tau):
    """ Solve SC step with BESSER
    """

    # Compute optical depth step
    tauMO = 0.5*(kk_m[0][0] + kk_o[0][0])*np.absolute((point_O.z - point_M.z)/np.cos(ray.rinc)) + c.vacuum
    tau += tauMO

    # Add to total tau
    tau_tot = np.append(tau_tot, tau_tot[-1] + tauMO[cdt.nus_N//2])

    # Compute exponentials
    exp_tauMO = np.empty(tauMO.shape)
    # Small linear
    small = (tauMO < 1e-7)
   #small = (tauMO < 1e-299)
    exp_tauMO[small] = 1. - tauMO[small] + 0.5*tauMO[small]*tauMO[small]
    # Normal
    normal = (tauMO >= 1e-7) & (tauMO < 300.)
   #normal = (tauMO < 300.)
    exp_tauMO[normal] = np.exp(-tauMO[normal])

    # Compute linear coeff
    psi_m, psi_o = psi_lin(exp_tauMO,tauMO)

   #print('tN',tauMO[9:12])
   #print('eN',exp_tauMO[9:12])
   #print('pmN',psi_m[9:12])
   #print('poN',psi_o[9:12])
   #print('wmN',wm)
   #print('woN',wo)
   #print('wcN',wc)
   #print('cmN',cm)

    # Build kappa and matrix
    kappa = []
    matri = []

    # First row
    kappa.append([])
    matri.append([])
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
    kappa[-1].append( psi_o*kk_o[0][1])
    kappa[-1].append( psi_o*kk_o[0][2])
    kappa[-1].append( psi_o*kk_o[0][3])
    matri[-1].append( exp_tauMO.copy())
    matri[-1].append(-psi_m*kk_m[0][1])
    matri[-1].append(-psi_m*kk_m[0][2])
    matri[-1].append(-psi_m*kk_m[0][3])
    # Second row
    kappa.append([])
    matri.append([])
   #kappa[-1].append( kappa[0][1].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
    kappa[-1].append( psi_o*kk_o[1][2])
    kappa[-1].append( psi_o*kk_o[1][1])
   #matri[-1].append( matri[0][1].copy())
    matri[-1].append( 0.)
   #matri[-1].append( exp_tauMO.copy())
    matri[-1].append( 1.)
    matri[-1].append(-psi_m*kk_m[1][2])
    matri[-1].append(-psi_m*kk_m[1][1])
    # Third row
    kappa.append([])
    matri.append([])
   #kappa[-1].append( kappa[0][2].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append(-kappa[1][2].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
    kappa[-1].append( psi_o*kk_o[1][0])
   #matri[-1].append( matri[0][2].copy())
    matri[-1].append( 0.)
    matri[-1].append(-matri[1][2].copy())
   #matri[-1].append( exp_tauMO.copy())
    matri[-1].append( 1.)
    matri[-1].append(-psi_m*kk_m[1][0])
    # Fourth
    kappa.append([])
    matri.append([])
   #kappa[-1].append( kappa[0][3].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append(-kappa[1][3].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append(-kappa[2][3].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
   #matri[-1].append( matri[0][3].copy())
    matri[-1].append( 0.)
   #matri[-1].append(-matri[1][3].copy())
    matri[-1].append( 0.)
   #matri[-1].append(-matri[2][3].copy())
    matri[-1].append( 0.)
   #matri[-1].append( exp_tauMO.copy())
    matri[-1].append( 1.)

    # Invert matrix
    kappa = matinv(kappa)

    # Matrix time vectors
    v1 = matrivec(matri,point_M.radiation.stokes)
    v2 = matvec(kappa,v1)

    # Quadratic
    if quad:

        # Compute optical depth step
        tauOP = 0.5*(kk_o[0][0] + kk_p[0][0])* \
                np.absolute((point_O.z - point_M.z)/np.cos(ray.rinc)) + c.vacuum

        # Compute BESSER coefficients
        wm,wo,wc = rt_omega(exp_tauMO,tauMO)

        # BESSER coefficient
        cm = BESSER_interp(tauMO, tauOP, sf_m, sf_o, sf_p)

        ss = [wm*sf_m[0] + wo*sf_o[0] +  wc*cm[0], \
              wm*sf_m[1] + wo*sf_o[1] +  wc*cm[1],
              wm*sf_m[2] + wo*sf_o[2] +  wc*cm[2],
              wm*sf_m[3] + wo*sf_o[3] +  wc*cm[3]]
        ss = matvec(kappa,ss)
        point_O.radiation.stokes = sumstkl(v2, ss)

    # Lineal
    else:

        ss = [psi_m*sf_m[0] + psi_o*sf_o[0], \
              psi_m*sf_m[1] + psi_o*sf_o[1], \
              psi_m*sf_m[2] + psi_o*sf_o[2], \
              psi_m*sf_m[3] + psi_o*sf_o[3]]
        ss = matvec(kappa, ss)
        point_O.radiation.stokes = sumstkl(v2, ss)

    return tau_tot


def psi_lin(ex,t):
    """ Compute linear contributions
    """

    big = t > 0.11
    small = t <= 0.11

    psi_m = np.empty(t.shape)
    psi_o = np.empty(t.shape)

    psi_m[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                    ((63e0 - 8e0*t[small])*t[small] - 432e0) + 2520e0) - \
                    12096e0) + 45360e0) - 120960e0) + 181440e0))/362880e0)
    psi_o[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                   ((9e0 - t[small])*t[small] - 72e0) + 504e0) - \
                    3024e0) + 15120e0) - 60480e0) + 181440e0))/362880e0)

    psi_m[big] = (1.-ex[big]*(1.+t[big]))/t[big]
    psi_o[big] = (ex[big]+t[big]-1.)/t[big]
   #psi_m = (1.-ex*(1.+t))/t
   #psi_o = (ex+t-1.)/t

    return psi_m,psi_o

def rt_omega(ex,t):
    """ Compute BESSER contributions
    """

    big = t > 0.14
    small = t <= 0.14

    omega_m = np.empty(t.shape)
    omega_o = np.empty(t.shape)
    omega_c = np.empty(t.shape)

    omega_m[big] = (2e0 - ex[big]*(t[big]*t[big] + 2e0*t[big] + 2e0))/(t[big]*t[big])
    omega_m[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                      ((140e0 - 18e0*t[small])*t[small] - \
                      945e0) + 5400e0) - 25200e0) + 90720e0) - \
                      226800e0) + 302400e0))/907200e0)

    big = t > 0.18
    small = t <= 0.18

    omega_o[big] = 1e0 - 2e0*(t[big] + ex[big] - 1e0)/(t[big]*t[big])
    omega_o[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                       ((10e0 - t[small])*t[small] - 90e0) + \
                       720e0) - 5040e0) + 30240e0) - 151200e0) + \
                       604800e0))/1814400e0)
    omega_c[big] = 2.0*(t[big] - 2.0 + ex[big]*(t[big] + 2.0))/(t[big]*t[big])
    omega_c[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                       ((35e0 - 4e0*t[small])*t[small] - \
                       270e0) + 1800e0) - 10080e0) + 45360e0) - \
                       151200e0) + 302400e0))/907200e0)

    return omega_m,omega_o,omega_c


def matinv(M):
    """ Inverts a matrix 4x4 with the absorption matrix symmetry properties\n
    """

    a = M[0][1]
    b = M[0][2]
    c = M[0][3]
    d = M[1][2]
    e = M[1][3]
    f = M[2][3]
    o = np.ones(a.shape)
    aa = a*a
    ab = a*b
    ac = a*c
    ad = a*d
    ae = a*e
    af = a*f
    bb = b*b
    bc = b*c
    bd = b*d
    be = b*e
    bf = b*f
    cc = c*c
    cd = c*d
    ce = c*e
    cf = c*f
    dd = d*d
    de = d*e
    df = d*f
    ee = e*e
    ef = e*f
    ff = f*f
    T0 = (af - be + cd)
    P1 = T0*a
    P2 = T0*b
    P3 = T0*c
    P4 = T0*d
    P5 = T0*e
    P6 = T0*f
    S1 = bc - de
    S2 = ac + df
    S3 = ab - ef
    S4 = ae + bf
    S5 = ad - cf
    S6 = bd + ce

    M[0][0] =   o + dd + ee + ff
    M[1][0] = - a + S6 - P6
    M[2][0] = - b - S5 + P5
    M[3][0] = - c - S4 - P4
    M[0][1] = - a - S6 - P6
    M[1][1] =   o - bb - cc + ff
    M[2][1] =   d + S3 - P3
    M[3][1] =   e + S2 + P2
    M[0][2] = - b + S5 + P5
    M[1][2] = - d + S3 + P3
    M[2][2] =   o - aa - cc + ee
    M[3][2] =   f + S1 - P1
    M[0][3] = - c + S4 - P4
    M[1][3] = - e + S2 - P2
    M[2][3] = - f + S1 + P1
    M[3][3] =   o - aa - bb + dd

    idet = o/(M[0][0] + a*M[1][0] + b*M[2][0] + c*M[3][0])
   
    for i in range(4):
        for j in range(4):
            M[i][j] *= idet

    return M

def matvec(A,b):
    """ Product matrix and vector with vector coefficients, for RT
    """

    c = [[],[],[],[]]

    c[0] = A[0][0]*b[0] + A[0][1]*b[1] + A[0][2]*b[2] + A[0][3]*b[3]
    c[1] = A[1][0]*b[0] + A[1][1]*b[1] + A[1][2]*b[2] + A[1][3]*b[3]
    c[2] = A[2][0]*b[0] + A[2][1]*b[1] + A[2][2]*b[2] + A[2][3]*b[3]
    c[3] = A[3][0]*b[0] + A[3][1]*b[1] + A[3][2]*b[2] + A[3][3]*b[3]

    return c

def matrivec(A,b):
    """ Product matrix and vector with vector coefficients, but it takes
        into account the symmetry properties of matri in BESSER, for RT
    """

    c = [[],[],[],[]]

    c[0] = A[0][0]*b[0] + A[0][1]*b[1] + A[0][2]*b[2] + A[0][3]*b[3]
    c[1] = A[0][1]*b[0] + A[0][0]*b[1] + A[1][2]*b[2] + A[1][3]*b[3]
    c[2] = A[0][2]*b[0] - A[1][2]*b[1] + A[0][0]*b[2] + A[2][3]*b[3]
    c[3] = A[0][3]*b[0] - A[1][3]*b[1] - A[2][3]*b[2] + A[0][0]*b[3]

    return c

def matmat(A,B):
    """ Product two 4x4 matrices
    """

    C = []

    # Row
    for ii in range(4):
        C.append([])
        for jj in range(4):
            C[-1].append(A[ii][jj]*B[jj][ii])

    return C

def sumstkl(a,b):
    """ Sum list of Stokes parameter contributions
    """

    c = []

    for i in range(4):
        c.append(a[i] + b[i])

    return c

def ybetwab(y,a,b):
    return (a <= b and y >= a and y <= b) or \
           (a >= b and y <= a and y >= b)

def correctyab(y,a,b):
    if b > a:
        mini = a
        maxi = b
    else:
        mini = b
        maxi = a

    if y <= mini and y <= maxi:
        return y
    elif y < mini:
        return mini
    else:
        return maxi

#@jit(nopython=True)
def BESSER_interp(tauMO, tauOP, sf_m, sf_o, sf_p):

    Cm = []

    # For Stokes
    for j in range(4):
        Cm.append(copy.copy(sf_o[j]))
        # For frequency
        for m, hm, hp, ym, yo, yp in zip(range(tauMO.size), tauMO, tauOP, sf_m[j],
                                         sf_o[j], sf_p[j]):

            # If both greater than 0
            if hm > 0. and hm > 0.:

                dm = (yo - ym)/hm
                dp = (yp - yo)/hp

            else:

                continue

            # If steps opposite sign
            if dm*dp <= 0.:
                continue

            # If same sign

            yder = (hm*dp + hp*dm)/(hm + hp)
            cm = yo - 0.5*hm*yder
            cp = yo + 0.5*hm*yder

            condm = ybetwab(cm, ym, yo)
            condp = ybetwab(cp, yo, yp)

            if condm and condp:
                Cm[j][m] = cm
            elif not condm:
                Cm[j][m] = correctyab(cm,ym,yo)
            elif not condp:
                cpp = correctyab(cp,yo,yp)
                yder = 2.0*(cpp - yo)/hp
                cm = yo - 0.5*hm*yder
                condpp = ybetwab(cm,ym,yo)

                if condpp:
                    Cm[j][m] = cm
                else:
                    Cm[j][m] = correctyab(cm,ym,yo)
    return Cm