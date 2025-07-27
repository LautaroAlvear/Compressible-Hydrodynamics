import sys
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#Returns T0,rho0 and p0
def Hydrostatic_Equilibrium(resolution=(128, 64, 96),dealias=3/2,R=1,Gamma=5/3):
    # Allow restarting via command line
    restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

    # Parameters
    Nphi, Ntheta, Nr = resolution
    R=1
    dtype = np.float64
    mesh = None
    dealias = 3/2

    Gamma = 5/3
    cv = 1
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
    ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=1, dealias=dealias, dtype=dtype)
    sphere = ball.surface

    # Fields
    p = dist.Field(name='p', bases=ball)
    T = dist.Field(name='T', bases=ball)
    Phi = dist.Field(name="phi",bases=ball)
    rho = dist.Field(bases=ball,name="rho")
    tau_Phi = dist.Field(bases=sphere,name="tau_Phi")

    phi, theta, r = dist.local_grids(ball)


    lift = lambda A: d3.Lift(A, ball, -1)
    p = 2*cv*rho*T/3                                    #p = R rho T/ mu
    problem = d3.LBVP(rho,Phi,)
