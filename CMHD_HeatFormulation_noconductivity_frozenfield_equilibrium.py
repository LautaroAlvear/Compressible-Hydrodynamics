import sys
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import math
from scipy.interpolate import interp1d
import sys

def safe_float(x):
    """Convierte un número a string reemplazando '.' por 'd'."""
    if isinstance(x, float):
        return str(round(x,3)).replace(".", "d")
    return str(x)

def leer_datos_y_interpolar(nombre_archivo):
    datos = np.loadtxt(nombre_archivo)
    interpoladores = {
        col: interp1d(datos[:, 0], datos[:, col], kind='linear', fill_value="extrapolate")
        for col in range(1, datos.shape[1])
    }
    def obtener(r, columna):
        if columna < 1 or columna >= datos.shape[1]:
            raise ValueError(f"Columna debe estar entre 1 y {datos.shape[1]-1}")
        return interpoladores[columna](r)
    return obtener

def crear_meshes_radiales(r_mesh, funcion_interp, Nphi, Ntheta):

    Nr = r_mesh.shape[2]
    r_vals = r_mesh[0,0,:]  # extraer perfil radial
    rho_radial = funcion_interp(r_vals, columna=1)  # rho(r)
    p_radial   = funcion_interp(r_vals, columna=2)  # p(r)
    Phi_radial = funcion_interp(r_vals, columna=3)  # Phi(r)

    rho_mesh = np.tile(rho_radial, (Nphi, Ntheta, 1))
    p_mesh   = np.tile(p_radial,   (Nphi, Ntheta, 1))
    Phi_mesh = np.tile(Phi_radial, (Nphi, Ntheta, 1))

    return rho_mesh, p_mesh, Phi_mesh

def rms(field):
    return d3.Integrate(d3.DotProduct(field,field)).evaluate()['g'][0][0][0]

def rms_slices(field):
    return d3.Integrate(d3.DotProduct(field,field))

def mag(field):
    return np.sqrt(d3.DotProduct(field,field))


#-----------------------Physical Parameters---------------------------#
nu = float(sys.argv[1])
kappa = float(sys.argv[2])
eta = float(sys.argv[3])
#nu, kappa, eta = 1,1,0

gamma = 5/3
Gamma = 5/3
Bstrength = np.sqrt(float(sys.argv[4]))


#-----------------------Simulation Parameters-------------------------#
Nphi, Ntheta, Nr = 1, 32, 48
R=1
dealias = 3/2
stop_sim_time = 50
timestepper = d3.SBDF2
max_timestep = 1e-5
dtype = np.float64
#mesh = (1,8)
mesh = None
it_writes = int(1e-2/max_timestep)                          #Writes every 1e-2 simulation times
rmsfile = f"RMS/rms_CMHD_equilibrium_frozenfield_{nu}_{kappa}_{eta}_{sys.argv[4]}.dat"
EnergyFile = f"Energies/Energies_CMHD_equilibrium_frozenfield_{nu}_{kappa}_{eta}_{sys.argv[4]}.dat"
filesnapshots = f"CMHD_equilibrium_frozenfield_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(float(sys.argv[4]))}_slices"



#--------------------------------------------------------------------------#
#------------------Object Initialization-----------------------------------#
#--------------------------------------------------------------------------#

#----Bases-----#
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=1, dealias=dealias, dtype=dtype)
sphere = ball.surface

#----Coordinates----#
phi, theta, r = dist.local_grids(ball)
phi_g, theta_g, r_g = ball.global_grids(dist,scales = ball.dealias)


#--------------------Fields------------------------------------#
u = dist.VectorField(coords, name='u',bases=ball)
B = dist.VectorField(coords,name="B",bases=ball)
S = dist.TensorField((coords,coords),name='S',bases=ball)
I = dist.TensorField((coords,coords),name='I',bases=ball)
p = dist.Field(name='p', bases=ball)
T = dist.Field(name='T', bases=ball)
Phi = dist.Field(name="Phi",bases=ball)
lnrho = dist.Field(bases=ball,name="lnrho")
grad_lnrho = d3.grad(lnrho) 

#----BC Fields----#
tau_u = dist.VectorField(coords, name='tau u', bases=sphere)
tau_T = dist.Field(name='tau T', bases=sphere)
tau_rho = dist.Field(bases=sphere,name="tau_rho")

#----Strain Tensor----#

I['g'][2][2] = 1
I['g'][1][1] = r**2
I['g'][0][0] = r**2*np.sin(theta)**2

strain = 1/2 * (d3.grad(u) + d3.trans(d3.grad(u)) )
expansion = 1/3*I*d3.Divergence(u)

strain_rate = d3.grad(u) + d3.trans(d3.grad(u))                  
shear_stress = d3.angular(d3.radial(strain_rate(r=1), index=1))

S = strain - expansion

S2 = d3.Trace(S@S.T)  

#---------------------------------------------------------#
#------------------Substitutions--------------------------#
#---------------------------------------------------------#

#----------------Auxiliary definitions-----------------#
er = dist.VectorField(coords, bases=ball.radial_basis)
er['g'][2] = 1
lift = lambda A: d3.Lift(A, ball, -1)
dr = lambda A: d3.DotProduct(d3.Gradient(A),er)

#-----Auxiliary Fields-----#
j = d3.Curl(B)
j2 = d3.DotProduct(j,j)
rho = np.exp(lnrho)
p = rho*T
cs = Gamma*p/rho

#----------Forces Definitions---------------#
PressureForce = -(Gamma-1)*d3.Gradient(p)/rho
GravForce = -d3.Gradient(Phi)
Advection = u@d3.Gradient(u)
ViscuousForce = 2*nu*d3.DotProduct(grad_lnrho,strain)-2/3*nu*grad_lnrho*d3.Divergence(u)+2*nu*d3.Divergence(strain)-2/3 *nu*d3.Gradient(d3.Divergence(u))
JcrossB = (d3.CrossProduct(j,B))/rho
ForceBalance = PressureForce+GravForce+Advection+ViscuousForce+JcrossB

#----------------------------------------------------------#
#-------------Initial Conditions---------------------------#
#----------------------------------------------------------#
aux_rho,aux_p,Phi['g'] = crear_meshes_radiales(r,leer_datos_y_interpolar("HydrostaticEquilibriumLaura.dat")
                        ,Nphi,Ntheta)

lnrho['g'] = np.log(aux_rho)
T['g'] = aux_p/aux_rho
T_boundary = T['g'][0][0][-1]
dT_boundary = dr(T).evaluate()['g'][0][0][-1]

constant = np.sqrt(4/3*np.pi)*Bstrength

B['g'][1] = -constant* np.sqrt(3/(4*np.pi*10.72727))*(8.75 - 21.*r**2 + 11.25*r**4) *np.sin(theta)
B['g'][2] = constant* np.sqrt(3/(4*np.pi*10.72727))*2*(4.375 - 5.25*r**2 + 1.875*r**4) *np.cos(theta)



#-------------------------------------------------------------------#
#---------------------Problem formulation---------------------------#
#-------------------------------------------------------------------#

problem = d3.IVP([lnrho, u, T, tau_u], namespace=locals())
#--------------Equations-----------------------------------#
problem.add_equation("dt(lnrho) + div(u) = -u@grad(lnrho) ")

problem.add_equation("""dt(u) - 2*nu*div(strain) + nu*(2/3)*grad(div(u)) + lift(tau_u) = -grad(Phi)+2*nu*dot(grad_lnrho, strain) - 2/3*nu*grad_lnrho*div(u) -(Gamma-1)*grad(p)/rho - u@grad(u)+cross(j,B)/rho""")

problem.add_equation("dt(T)  = - u@grad(T)-(Gamma-1)*T*div(u)+2*nu*S2")

#--------------Boundary Conditions------------------------#
problem.add_equation("shear_stress = 0")  # Stress free
problem.add_equation("radial(u(r=R)) = 0")  # No penetration

#----------------Solver-------------------#
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

initial_timestep = max_timestep
file_handler_mode = 'overwrite'

#-----------------------------------------------------------------------#
#----------------------Analysis and Post-Processing---------------------#
#-----------------------------------------------------------------------#

slices = solver.evaluator.add_file_handler(filesnapshots, sim_dt=0.001, max_writes=10000, mode=file_handler_mode)

#---------------------------Problem variables-------------------------#
slices.add_task(u,scales=dealias,name="u",layout="g")
slices.add_task(np.sqrt(d3.DotProduct(u,u)),scales=dealias,name="magu",layout="g")
slices.add_task(d3.DotProduct(u,u),scales=dealias,name="u2")
slices.add_task(T,scales=dealias,name="T",layout="g")
slices.add_task(lnrho,scales=dealias,name="lnrho",layout="g")
slices.add_task(cs,scales=dealias,name="cs")
slices.add_task(rho,layout="g",name="rho")
slices.add_task(p,layout="g",name="p")

#----------------------------Spectrum-----------------------------#
slices.add_task(T,scales=dealias,name="T_coef",layout="c")
slices.add_task(lnrho,scales=dealias,name="lnrho_coef",layout="c")

#----------------------------Tau terms-----------------------------#
slices.add_task(tau_u,name="tau_u",layout="g")
slices.add_task(tau_u,name="tau_u_coeff",layout="c")

#------------------------------Forces-------------------------------#
slices.add_task(PressureForce,name="PressureForce",layout="g")
slices.add_task(Advection,name="Advection",layout="g")
slices.add_task(GravForce,name="GravitationalForce",layout="g")
slices.add_task(ViscuousForce,name="ViscuousForce",layout="g")
slices.add_task(ForceBalance,name = "ForceBalance",layout="g")
slices.add_task(JcrossB,name="JcrossB",layout='g')

#-------Magnitudes--------#
slices.add_task(mag(PressureForce),name="magPressureForce",layout="g")
slices.add_task(mag(Advection),name="magAdvection",layout="g")
slices.add_task(mag(GravForce),name="magGravitationalForce",layout="g")
slices.add_task(mag(ViscuousForce),name="magViscuousForce",layout="g")
slices.add_task(mag(ForceBalance),name = "magForceBalance",layout="g")
slices.add_task(mag(JcrossB),name="magJcrossB",layout='g')

#----------------------------Heating Terms-------------------------#
slices.add_task(2*nu*S2,name="ViscousHeating",layout='g')
slices.add_task(-d3.DotProduct(u,d3.Gradient(T)),name="HeatTransport",layout='g')
slices.add_task(-(Gamma-1)*T*d3.Divergence(u),name="DensityHeating",layout='g')
slices.add_task(2*nu*S2-d3.DotProduct(u,d3.Gradient(T))-(Gamma-1)*T*d3.Divergence(u),name="HeatingBalance",layout='g')


#---------------------------RMS Quantities-------------------------#

#--------Forces---------#
slices.add_task(rms_slices(PressureForce),name="RMSPressureForce",layout="g")
slices.add_task(rms_slices(Advection),name="RMSAdvection",layout="g")
slices.add_task(rms_slices(GravForce),name="RMSGravitationalForce",layout="g")
slices.add_task(rms_slices(ViscuousForce),name="RMSViscuousForce",layout="g")
slices.add_task(rms_slices(ForceBalance),name = "RMSForceBalance",layout="g")
slices.add_task(rms_slices(JcrossB),name="RMSJcrossB",layout='g')


#-----Heating Termns----#
slices.add_task(d3.Integrate(2*nu*S2),name="RMSViscousHeating",layout='g')
slices.add_task(d3.Integrate(d3.DotProduct(u,d3.Gradient(T))),name="RMSHeatTransport",layout='g')
slices.add_task(d3.Integrate((Gamma-1)*T*d3.Divergence(u)),name="RMSDensityHeating",layout='g')
slices.add_task(d3.Integrate(2*nu*S2-d3.DotProduct(u,d3.Gradient(T))-(Gamma-1)*T*d3.Divergence(u)),name="RMSHeatingBalance",layout='g')


#---------------------------Energies-------------------------------#
slices.add_task(0.5*d3.Integrate(rho*d3.DotProduct(u,u)),name="KineticEnergy",layout='g')
slices.add_task(0.5*d3.Integrate(rho*Phi),name="GravEnergy",layout='g')
slices.add_task(0.5*d3.Integrate(d3.DotProduct(B,B)),name="MagneticEnergy",layout='g')
slices.add_task(0.5*d3.Integrate(rho*T),name="InternalEnergy",layout='g')
slices.add_task(0.5*d3.Integrate(rho*d3.DotProduct(u,u))+0.5*d3.Integrate(rho*Phi)+0.5*d3.Integrate(d3.DotProduct(B,B))+0.5*d3.Integrate(rho*T)
                ,name="TotalEnergy",layout='g')
slices.add_task(d3.Integrate(rho),name="Mass",layout='g')





checkpoints = solver.evaluator.add_file_handler('checkpoints', iter=it_writes, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# CFL
CFL = d3.CFL(solver, initial_timestep, cadence=10, safety=0.5, threshold=0.1, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')
flow.add_property(cs,name="cs")

with open(rmsfile,'w') as file:
    file.write("#Time\tPressure\t Grav \t Advection\tViscuous\tBalance\tJcrossB\n")

with open(EnergyFile,'w') as file:
    file.write("#Time\tKin.\tGrav\tInt\tMagnetic\tMass\n")





# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e" %(solver.iteration, solver.sim_time, timestep, max_u))
            if math.isnan(max_u):
                exit()
        if solver.iteration % 1000 == 0:
            with open(rmsfile,"a") as file:
                file.write(f"{solver.sim_time}\t{rms(PressureForce)}\t{rms(GravForce)}\t{rms(Advection)}\t{rms(ViscuousForce)}\t{rms(JcrossB)}\t{rms(ForceBalance)}\n")

            with open(EnergyFile,'a') as file:
                file.write(f"{solver.sim_time}\t{0.5*d3.Integrate(rho*d3.DotProduct(u,u)).evaluate()['g'][0][0][0]}\t{0.5*d3.Integrate(rho*Phi).evaluate()['g'][0][0][0]}\t{0.5*d3.Integrate(rho*T).evaluate()['g'][0][0][0]}\t{0.5*d3.Integrate(d3.DotProduct(B,B)).evaluate()['g'][0][0][0]}\t{d3.Integrate(rho).evaluate()['g'][0][0][0]}\n")
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
