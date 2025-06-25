#!/usr/bin/env python
# coding: utf-8

# # Preliminary Experiments for Compressible Hydrodynamics
# 
# Before jumping to the more advanced simulation of **Compressible Magnetohydrodynamics (CMHD)** I begin with studying the capacity of Dedalus3 to solve the compressible hydrodynamics, since the term that generate shocks $$\frac{\partial \ln \rho}{\partial t} = -\nabla \cdot \mathbf{u} - \mathbf{u} \cdot \nabla (\ln \rho) $$
# might be unsolvable for Dedalus. This is due to shocks needing for high resolution, which can lead to an spectrum that does not converge.
# 
# Following the discussions in the Dedalus forum, I chose to solve for $\ln \rho$.
# 
# The equations to be solved are
# \begin{align*}
# \frac{\partial \ln \rho}{\partial t} &= -\nabla \cdot \mathbf{u} - \mathbf{u} \cdot \nabla (\ln \rho) \\
# \frac{\partial \mathbf{u}}{\partial t} &= -\mathbf{u} \cdot \nabla \mathbf{u} - \frac{\nabla p}{\rho} - \nabla \Phi + \frac{\mathbf{j \times B}}{\rho} + \frac{\nabla \cdot (2\rho \nu \mathbf{S})}{\rho}\\
# \frac{\partial s}{\partial t} &= - \mathbf{u} \cdot \nabla s + \frac{\eta \mu_0 |\mathbf{j} |^2}{\rho T} + \frac{2 \nu S^2}{T} 
# \end{align*}
# 
# As for boundary conditions, I'm not sure as to which ones are appropiate for the problem to have physical relevance, but Nicolas suggested **no stress boundary conditions on the shell**. Since the outter region is a vacuum, I set a very low density in the shell $$\rho(r=R) = 10^{-4} \qquad \qquad \ln \rho (r=R) = -9.21034$$ 
# And for simplicity I impose the boundary condition $$s(r=R) = 0$$
# 
# **Questions:**
# In most of the literature and even Wikipedia, I see that the rate of shear tensor doesn't have the term $$-\frac{2}{3} \delta_{ij} \nabla \cdot \mathbf{u}$$

# In[1]:


####------------------------Code parameters and Definitions----------------------#
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from scipy.special import erf
#shutil.rmtree('snapshots', ignore_errors=True)
from mpl_toolkits.mplot3d import Axes3D
import logging
logger = logging.getLogger(__name__)
from scipy.interpolate import griddata
import matplotlib.colors as clr



#---------------------------Parameters------------------------------------#
Nphi, Ntheta, Nr = 128, 64, 96
dealias = 1                                             #Por ahora fijo en 1 porque da problemas si no.
dtype = np.float64
mesh = None
R = 1
mapa2 = list(zip([0,.4,.5,.6,1.0], ["brown","yellow","gainsboro","aqua", "mediumblue"]))
cmapchi = clr.LinearSegmentedColormap.from_list("", mapa2, N=256) 
eta = 1e-10
sc,Tc,rho_c = 1e4, 1e3, 1e2
Gamma = 5/3
cv = 100
mu = 0.6
mu_0 = 0.001
nu = 0.021



#--------------------------Funcion auxiliar para graficar-------------------#
def plot_mapa_calor(R, Theta, colormesh, ax,label,fig,plotcolorbar=True,cmap='PRGn',diverging=False,polar=True,vmin=0,vmax=100,autov=True):
    if diverging:
        lista_max = [max(colormesh[i]) for i in range(len(colormesh))]
        vmax = max(lista_max)
        pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap=cmap,vmax=vmax,vmin=-vmax)
    else:
        if not autov:
            pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap=cmap)
    if polar:
        ax.set_theta_zero_location('N')       # Cero en el norte (arriba)
        ax.set_theta_direction(-1)  
        ax.set_thetamax(180)
        ax.set_yticks([])
        ax.set_xticks([])

    if(plotcolorbar):
        fig.colorbar(pcm, ax=ax, orientation='vertical', label=label)  

def plot_stream_w_mag(R,Theta,F_r,F_theta,colormesh,fig,ax,label="",cmap="plasma",scale="log",vmin=-100,vmax=100,autov=True,density=1.5):

    X = R*np.sin(Theta)
    Y = R*np.cos(Theta)

    F_x = F_r*np.sin(Theta)+F_theta*np.cos(Theta)
    F_y = F_r*np.cos(Theta)-F_theta*np.sin(Theta)


    x = X.flatten()
    y = Y.flatten()
    u = F_x.flatten()
    v = F_y.flatten()
    color = colormesh.flatten()

    #regular grid
    x_aux = np.linspace(x.min(), x.max(), 500)
    y_aux = np.linspace(y.min(), y.max(), 500)
    X_regular, Y_regular = np.meshgrid(x_aux, y_aux)

    # Interpolar
    U_regular = griddata((x, y), u, (X_regular, Y_regular), method='cubic')
    V_regular = griddata((x, y), v, (X_regular, Y_regular), method='cubic')
    color_interpolated = griddata((x, y), color, (X_regular, Y_regular), method='cubic')
    
    ax.streamplot(X_regular, Y_regular, U_regular, V_regular, density=density, linewidth=0.7, arrowsize=1,color="black")
    ax.plot(np.cos(np.linspace(0, 2*np.pi, 300)),
             np.sin(np.linspace(0, 2*np.pi, 300)),
            'k', linewidth=2)

    if autov:
        pcm = ax.pcolormesh(X_regular,Y_regular,color_interpolated,cmap=cmap,shading='nearest',norm=scale)
    else:
        pcm = ax.pcolormesh(X_regular,Y_regular,color_interpolated,cmap=cmap,shading='nearest',norm=scale,vmin=vmin,vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=label, pad=.1)
    ax.set_xlim(0,1)
    ax.set_ylim(-1.1,1.1)
    ax.set_aspect('equal')
    plt.tight_layout()


# In[9]:


#-----------------------------------Object Initialization---------------------------------#

coords = d3.SphericalCoordinates("phi","theta","r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=R, dealias=dealias, dtype=dtype)
sphere = ball.surface

#--------Fields---------------------------#
B = dist.VectorField(coords,bases=ball,name="B")
S = dist.TensorField((coords,coords),bases=ball,name="S")
I = dist.TensorField((coords,coords),bases=ball,name="I")

u = dist.VectorField(coords,bases=ball,name='u')
lnrho = dist.Field(bases=ball,name="lnrho")
s = dist.Field(bases=ball,name="s")
tau_u = dist.VectorField(coords,bases=sphere,name="tau_u")
tau_rho = dist.Field(bases=sphere,name="tau_rho")
tau_s = dist.Field(bases=sphere,name="tau_s")

#-----------Sustitutions-----------------#
phi, theta, r = dist.local_grids(ball)
phi_g, theta_g, r_g = ball.global_grids(dist,scales = ball.dealias)
lift_basis = ball.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
radial = lambda A: d3.RadialComponent(A)
cross = lambda A,B:d3.CrossProduct(A,B)
rho = np.exp(lnrho)
dr = lambda A: d3.RadialComponent(d3.Gradient(A))
T = Tc*(rho/rho_c)**(Gamma-1)*np.exp((s-sc)/cv)
p = R*rho*T/mu


#--------------------------------------Initial Conditions---------------------------------#

constant = np.sqrt(4/3*np.pi)
#phi, theta, r = ball.global_grids(dist,scales = ball.dealias)
#-----Magnetic field Model 1------#
#B['g'][1] = -constant* np.sqrt(3/(4*np.pi*10.72727))*(8.75 - 21.*r**2 + 11.25*r**4) *np.sin(theta)
#B['g'][2] = constant* np.sqrt(3/(4*np.pi*10.72727))*2*(4.375 - 5.25*r**2 + 1.875*r**4) *np.cos(theta)
j= d3.Curl(B)
j2 = d3.DotProduct(j,j)


lnrho['g'] = np.log(np.exp(-(r-0.5)**2 /0.001))
plt.plot(r[0][0],rho['g'][0][0])
plt.show()
s.fill_random('g')
rho = np.exp(lnrho)



I['g'][2,2] = 1
I['g'][1,1] = r**2
I['g'][0,0] = r**2*np.sin(theta)**2
S = d3.grad(u) + d3.trans(d3.grad(u))
#+ I*d3.Divergence(u)
shear_stress = d3.angular(d3.radial(S(r=1), index=1))

S2 = d3.Trace(S@S.T)              


fb = d3.CrossProduct(d3.Curl(B),B)


# In[ ]:


import math
#--------------------Problem Formulation-------------------------------#
problem = d3.IVP([u,tau_u,lnrho,tau_rho,s,tau_s],namespace=locals())

problem.add_equation("div(u)+dt(lnrho)+lift(tau_rho) = u@grad(lnrho)")
problem.add_equation("dt(u)  + lift(tau_u) = fb/rho-u@grad(u)-grad(p)/rho+1/rho*div(S*2*rho*eta)")
problem.add_equation("dt(s) + lift(tau_s) = -u@grad(s)+(eta*mu_0*j2)/(rho*T) + 2*nu*S2/T")
problem.add_equation("radial(u(r=R)) = 0")                #No penetration
problem.add_equation("shear_stress = 0")                  #No stress
problem.add_equation("lnrho(r=R) = -9.21034")             #Vacuum outside
problem.add_equation("s(r=R) = 10000")                    
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 20

snapshots = solver.evaluator.add_file_handler(f'snapshots', sim_dt=.001, max_writes=30000,iter=10)
snapshots.add_task(u,name='u',layout='g')
snapshots.add_task(lnrho,name='lnrho',layout='g')
snapshots.add_task(s,name='s',layout='g')
snapshots.add_task(T,name='T',layout='g')


max_timestep = 0.05
initial_timestep = max_timestep

# CFL
CFL = d3.CFL(solver, initial_timestep, cadence=10, safety=0.5, threshold=0.1, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e" %(solver.iteration, solver.sim_time, timestep, max_u))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

# In[ ]:
exit()

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import h5py
import matplotlib.animation as manimation
import shutil
from scipy.special import erf
from matplotlib.colors import ListedColormap

def Animacion_Mapacalor(taskname,name,titulo,minimo=-1,maximo=1,cmap="hot"):
    
    with h5py.File("snapshots/snapshots_s1.h5", mode='r') as file:
        # Load datasets
        lapu = file['tasks'][taskname]
        t = lapu.dims[0]['sim_time']
        phi = lapu.dims[1][0]        
        theta = lapu.dims[2][0]
        r = lapu.dims[3][0]
        
        Theta, R = np.meshgrid(theta, r)
        step = 1

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Plotter', artist='lau',
                    comment='')
        writer = FFMpegWriter(fps=10, metadata=metadata)
        print(np.shape(lapu))

        fig,(ax) = plt.subplots(1,1,subplot_kw={'projection':'polar'})


        with writer.saving(fig, f"{name}.mp4", 100):
            for i in range(0,len(lapu),step):
                print(f"it {i} de {len(lapu)}")
                colormesh = lapu[i][0]
                print(np.shape(colormesh))
                

                pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap=cmap)

                ax.set_title(titulo + f"\nt={round(t[i],2)}")
                cb = fig.colorbar(pcm, ax=ax, orientation='vertical', label=name)
                ax.set_theta_zero_location('N')       # Cero en el norte (arriba)
                ax.set_theta_direction(-1)  
                ax.set_thetamax(180)
                ax.set_yticks([])
                ax.set_xticks([])
                writer.grab_frame()
                cb.remove()
                ax.cla()

Animacion_Mapacalor("s","s","Entropy")

