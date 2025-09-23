
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import h5py
import matplotlib.animation as manimation
import shutil
from scipy.special import erf
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from matplotlib.animation import PillowWriter

def safe_float(x):
    """Convierte un número a string reemplazando '.' por 'd'."""
    if isinstance(x, float):
        return str(round(x,3)).replace(".", "d")
    return str(x)

def Temperature_Plot(lapu,t,R,Theta,ax,fig,path,viscosities,writer,time_index=-1,max_colorbar=None):

    colormesh = lapu[time_index][0]
    if max_colorbar == None:
        pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap="magma")

    else:
        pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap="magma",vmax=max_colorbar)


    ax.set_title(f"t={round(t[time_index],3):.3f}"+r"$t_{G}$"+"\n"+r"$\nu$=" + viscosities[0]+"\t\t"+r"$\kappa = $"+viscosities[1]+"\n"+r"$\eta =$" + viscosities[2])


    cb = fig.colorbar(pcm, ax=ax, orientation='vertical', label="T")
    cb.set_label(fontsize=14,label="T")
    ax.set_theta_zero_location('N')       # Cero en el norte (arriba)
    ax.set_theta_direction(-1)  
    ax.set_thetamax(180)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    writer.grab_frame()
    cb.remove()
    ax.cla()

def Density_plot(lapu,t,R,Theta,ax,fig,path,viscosities,writer,time_index=-1,max_colorbar=None):
            
    colormesh = lapu[time_index][0]

    if max_colorbar == None:
        pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap="bone")

    else:
        pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap="bone",vmax=max_colorbar)
    
    
    ax.set_title(f"t={round(t[time_index],3):.3f}"+r"$t_{G}$"+"\n"+r"$\nu$=" + viscosities[0]+"\t\t"+r"$\kappa = $" +viscosities[1]+"\t\t"+r"$\eta =$" + viscosities[2])
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical', label="T")
    cb.set_label(fontsize=14,label=r"$\rho$")
    ax.set_theta_zero_location('N')       # Cero en el norte (arriba)
    ax.set_theta_direction(-1)  
    ax.set_thetamax(180)
    ax.set_yticks([])
    ax.set_xticks([])
    writer.grab_frame()
    cb.remove()
    ax.cla()

def ArbitraryTask_Plot(lapu,t,R,Theta,ax,fig,path,viscosities,writer,time_index=-1,max_colorbar=None,tasklabel="ViscousHeating"):

    colormesh = lapu[time_index][0]
    if max_colorbar == None:
        pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap="magma")

    else:
        pcm = ax.pcolormesh(Theta,R,np.transpose(colormesh), shading="nearest", cmap="magma",vmax=max_colorbar)


    ax.set_title(f"t={round(t[time_index],3):.3f}"+r"$t_{G}$"+"\n"+r"$\nu$=" + viscosities[0]+"\t\t"+r"$\kappa = $" +viscosities[1]+"\t\t"+r"$\eta =$" + viscosities[2])


    cb = fig.colorbar(pcm, ax=ax, orientation='vertical', label="T")
    cb.set_label(fontsize=14,label=tasklabel)
    ax.set_theta_zero_location('N')       # Cero en el norte (arriba)
    ax.set_theta_direction(-1)  
    ax.set_thetamax(180)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    writer.grab_frame()
    cb.remove()
    ax.cla()

def Animacion_Mapacalor(path,viscosities,name="plot",plot_type="Density",step=1,automax=False):

    plot_dict = {"Density":Density_plot,"Temperature":Temperature_Plot,"ViscousHeating":ArbitraryTask_Plot,
                 "DensityHeating":ArbitraryTask_Plot,"HeatTransport":ArbitraryTask_Plot,"Pressure":ArbitraryTask_Plot,
                 'Sound':ArbitraryTask_Plot,'Velocity':ArbitraryTask_Plot}
    colormesh_dict = {"Density":"rho","Temperature":"T","DensityHeating":"DensityHeating","ViscousHeating":"ViscousHeating",
                      "HeatTransport":"HeatTransport","Pressure":"p","Sound":"cs","Velocity":"magu"}
    tasklabel_dict = {"HeatTransport":r"$u\cdot \nabla T$","DensityHeating":r"$(\Gamma-1)T\nabla \cdot u$","ViscousHeating":r"$2\nu S^2$",
                      "Density":r"$\rho$","Temperature":"T","Pressure":r"$p$","Sound":r"$c_s$","Velocity":r"$\left |\mathbf{u} \right |$"}

    plot_func = plot_dict[plot_type]
    taskname = colormesh_dict[plot_type]
    tasklabel = tasklabel_dict[plot_type]

    
    
    fig,(ax) = plt.subplots(1,1,subplot_kw={'projection':'polar'})
    fps = 20
    FFMpegWriter = PillowWriter
    metadata = dict(title='Plotter', artist='lau',
                        comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, f"{name}.gif", 100):

        with h5py.File(f"{path}", mode='r') as file:
            # Load datasets
            lapu = file['tasks'][taskname]
            t = lapu.dims[0]['sim_time']
            print(np.max(t))
            phi = lapu.dims[1][0]        
            theta = lapu.dims[2][0]
            r = lapu.dims[3][0]

            if automax:
                max_colorbar=max(lapu[:].flatten())
            else:
                max_colorbar = None

            print(max_colorbar)
                
            Theta, R = np.meshgrid(theta, r)
            
            for i in range(1*fps):
                if plot_func == ArbitraryTask_Plot:
                    plot_func(lapu,t,R,Theta,ax,fig,path,viscosities,writer,0,max_colorbar=max_colorbar,tasklabel=tasklabel)
                else:
                    plot_func(lapu,t,R,Theta,ax,fig,path,viscosities,writer,0,max_colorbar=max_colorbar)
            
            for i in range(step,len(lapu),step):
                if plot_func == ArbitraryTask_Plot:
                    plot_func(lapu,t,R,Theta,ax,fig,path,viscosities,writer,i,max_colorbar=max_colorbar,tasklabel=tasklabel)
                else:
                    plot_func(lapu,t,R,Theta,ax,fig,path,viscosities,writer,i,max_colorbar=max_colorbar)


def VectorField_Plot(F_r,F_theta,colormesh,t,R,Theta,ax,fig,path,viscosities,writer,time_index=-1,max_colorbar=None,tasklabel="ViscousHeating",density=1.5):


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

    pcm = ax.pcolormesh(X_regular,Y_regular,color_interpolated,cmap="Purples",shading='nearest')
    cb = fig.colorbar(pcm, ax=ax, label=tasklabel, pad=.1)

    ax.set_xlim(0,1)
    ax.set_ylim(-1.1,1.1)
    ax.set_aspect('equal')
    ax.set_title(f"t={round(t[time_index],3):.3f}"+r"$t_{G}$"+"\n"+r"$\nu$=" + viscosities[0]+"\t\t"+r"$\kappa = $"+viscosities[1]+"\n"+r"$\eta =$" + viscosities[2])
    ax.set_xticks([])
    ax.set_yticks([])
    writer.grab_frame()
    cb.remove()
    ax.cla()
    
    pass

def Animacion_StreamPlot(path,viscosities,name="plot",plot_type="Density",step=1,automax=True):

    streamplot_dict = {"Velocity":VectorField_Plot,"PressureForce":VectorField_Plot,"ViscousForce":VectorField_Plot,"JcrossB":VectorField_Plot,
                       "ForceBalance":VectorField_Plot,"S2":VectorField_Plot}
    colormesh_dict = {"Velocity":"magu","PressureForce":"magPressureForce","ViscousForce":"magViscuousForce","JcrossB":"magJcrossB","ForceBalance":"magForceBalance",
                      "S2":"magS2"}
    vectormesh_dict = {"Velocity":"u","PressureForce":"PressureForce","ViscousForce":"ViscuousForce","JcrossB":"JcrossB","ForceBalance":"ForceBalance",
                       "S2":"S2"}
    tasklabel_dict = {"Velocity":r"$\left |\mathbf{u} \right |$","PressureForce":r"$-\nabla p/\rho$","ViscousForce":r"$2\nu\nabla \cdot \left (  \matbf{S}^2 \rho \right )/\rho$",
                      "JcrossB":r"$\left ( \mathbf{J \times B} \right )/\rho$","ForceBalance":"Force Balance"}

    streamplot_func = streamplot_dict[plot_type]
    colortaskname = colormesh_dict[plot_type]
    vectortaskname = vectormesh_dict[plot_type]
    tasklabel = tasklabel_dict[plot_type]

    
    
    fig,(ax) = plt.subplots(1,1)
    fps = 10
    FFMpegWriter = PillowWriter
    metadata = dict(title='Plotter', artist='lau',
                        comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, f"{name}.gif", 100):

        with h5py.File(f"{path}", mode='r') as file:
            print("AA")
            # Load datasets
            u = file['tasks'][vectortaskname]
            magu = file['tasks'][colortaskname]
            t = u.dims[0]['sim_time']
            
            phi = u.dims[2][0]        
            theta = u.dims[3][0]
            r = u.dims[4][0]
            Theta, R = np.meshgrid(theta, r,indexing='ij')
            
            F_r = [i[2][0] for i in u ]
            F_theta = [i[1][0] for i in u ]
            coloru = [i[0] for i in magu]

            if automax:
                max_colorbar=max(u[:].flatten())
            else:
                max_colorbar = None

            print(max_colorbar)
            
            for i in range(1*fps):
                if streamplot_func == VectorField_Plot:
                    streamplot_func(F_r[0],F_theta[0],coloru[i],t,R,Theta,ax,fig,path,viscosities,writer,0,tasklabel=tasklabel)
                else:
                    streamplot_func(F_r,F_theta[i],coloru[i],t,R,Theta,ax,fig,path,viscosities,writer,1,tasklabel=tasklabel)
            
            for i in range(step,len(u),step):
                if streamplot_func == VectorField_Plot:
                    streamplot_func(F_r[i],F_theta[i],coloru[i],t,R,Theta,ax,fig,path,viscosities,writer,i,tasklabel=tasklabel)
                else:
                    streamplot_func(F_r,F_theta,coloru,t,R,Theta,ax,fig,path,viscosities,writer,i,tasklabel=tasklabel)



def ScalarPlot(path,task_list,labels_list,cmap="Dark2",linestyles=None,colors_list=None,name="Plot.pdf"):
    
    dict_index = dict(zip(task_list,range(len(task_list))))
    dict_labels = dict(zip(task_list,labels_list))
   
    if colors_list==None:
        color =  plt.get_cmap(cmap)
        dict_colors = dict(zip(task_list,[color(dict_index[i]/len(task_list)) for i in task_list]))
    else:
        dict_colors = dict(zip(task_list,colors_list))
    if linestyles == None:
        dict_linestyles = dict(zip(task_list,["-" for i in task_list]))
    
    else:
        dict_linestyles = dict(zip(task_list,linestyles))
    
    fig, ax = plt.subplots(1,1)
    
    for i in task_list:
        print(i)
        with h5py.File(f"{path}", mode='r') as file:
            task = file['tasks'][i]
            t = task.dims[0]['sim_time']
            plotaux = task[:].reshape(len(t[:]))
            
            max_y = np.percentile(plotaux, 98)

            ax.plot(t[:],plotaux,dict_linestyles[i],label=dict_labels[i],color=dict_colors[i])
    
    ax.set_xlabel("t",fontsize=14)
    ax.legend()
    ax.set_ylim(top=max_y)
    plt.savefig(name)




def FullPlotCMHD(simtype,nu,kappa,eta,Bstrength,it_step=5,PlotEnergies=True,PlotRMS=True,PlotStreamplots=True,PlotHeatmaps=True):
    
    #------------------------------Mapas de Calor------------------------#
    if PlotHeatmaps:
        Animacion_Mapacalor(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                            (str(nu),str(kappa),str(nu)),f"rho_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}",step=it_step,automax=False)
        
        Animacion_Mapacalor(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                            (str(nu),str(kappa),str(nu)),f"T_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","Temperature",step=it_step,automax=False)

        Animacion_Mapacalor(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                            ("1","0","0","0.1"),f"p_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","Pressure",step=it_step)
        
        Animacion_Mapacalor(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                            ("1","0","0","0.1"),f"cs_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","Sound",step=it_step)
        
        Animacion_Mapacalor(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                            ("1","0","0","0.1"),f"magu_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","Velocity",step=it_step)

    #-------------------------Cantidades RMS--------------------------------#
    if PlotRMS:
        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                ["RMSViscousHeating",'RMSHeatTransport',"RMSDensityHeating",'RMSHeatingBalance'],["ViscousHeating",'HeatTransport',"DensityHeating",'HeatingBalance'],
                name=f"HeatingTerms_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}.pdf")

        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                ["RMSPressureForce",'RMSGravitationalForce',"RMSJcrossB",'RMSViscuousForce','RMSForceBalance','RMSAdvection'],
                ["PressureForce",'GravitationalForce',"JcrossB",'ViscousForce','ForceBalance','Advection'],
                name=f"RMSForces_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}.pdf")
    if PlotEnergies:
        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",
                ["KineticEnergy",'GravEnergy',"MagneticEnergy",'InternalEnergy'],
                ["Kinetic",'Gravitational',"Magnetic",'Internal'],name=f"Energies_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}.pdf")

        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",['Mass'],
                ['Mass'],name=f"Conservations_CMHD_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}.pdf")
        
    #---------------------------StreamPlots-----------------------------------#
    if PlotStreamplots:
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","Velocity",step=10)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","PressureForce",step=10)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","ViscousForce",step=10)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","JcrossB",step=10)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_{safe_float(eta)}_{safe_float(Bstrength)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_{eta}_{Bstrength}","ForceBalance",step=10)
    


def FullPlotCHD(simtype,nu,kappa,it_step=5,PlotEnergies=True,PlotRMS=True,PlotStreamplots=True,PlotHeatmaps=True):
    
    #------------------------------Mapas de Calor------------------------#
    if PlotHeatmaps:
        Animacion_Mapacalor(f"slices/CHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                            (str(nu),str(kappa),str(nu)),f"rho_CHD_{simtype}_{nu}_{kappa}_",step=it_step,automax=False)
        
        Animacion_Mapacalor(f"slices/CHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                            (str(nu),str(kappa),str(nu)),f"T_CHD_{simtype}_{nu}_{kappa}_","Temperature",step=it_step,automax=False)

        Animacion_Mapacalor(f"slices/CHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                            ("1","0","0","0.1"),f"p_CHD_{simtype}_{nu}_{kappa}_","Pressure",step=it_step)
        
        Animacion_Mapacalor(f"slices/CHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                            ("1","0","0","0.1"),f"cs_CHD_{simtype}_{nu}_{kappa}_","Sound",step=it_step)
        
        Animacion_Mapacalor(f"slices/CHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                            ("1","0","0","0.1"),f"magu_CHD_{simtype}_{nu}_{kappa}_","Velocity",step=it_step)

    #-------------------------Cantidades RMS--------------------------------#
    if PlotRMS:
        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                ["RMSViscousHeating",'RMSHeatTransport',"RMSDensityHeating",'RMSHeatingBalance'],["ViscousHeating",'HeatTransport',"DensityHeating",'HeatingBalance'],
                name=f"HeatingTerms_CHD_{simtype}_{nu}_{kappa}_.pdf")

        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                ["RMSPressureForce",'RMSGravitationalForce',"RMSJcrossB",'RMSViscuousForce','RMSForceBalance','RMSAdvection'],
                ["PressureForce",'GravitationalForce',"JcrossB",'ViscousForce','ForceBalance','Advection'],
                name=f"RMSForces_CHD_{simtype}_{nu}_{kappa}_.pdf")
    if PlotEnergies:
        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",
                ["KineticEnergy",'GravEnergy',"MagneticEnergy",'InternalEnergy'],
                ["Kinetic",'Gravitational',"Magnetic",'Internal'],name=f"Energies_CHD_{simtype}_{nu}_{kappa}_.pdf")

        ScalarPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",['Mass'],
                ['Mass'],name=f"Conservations_CHD_{simtype}_{nu}_{kappa}_.pdf")
        
    #---------------------------StreamPlots-----------------------------------#
    if PlotStreamplots:
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_","Velocity",step=it_step)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_","PressureForce",step=it_step)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_","ViscousForce",step=it_step)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_","JcrossB",step=it_step)
        
        Animacion_StreamPlot(f"slices/CMHD_{simtype}_{safe_float(nu)}_{safe_float(kappa)}_slices_s1.h5",("1","0","0","0.1"),
                            f"u_{simtype}_{nu}_{kappa}_","ForceBalance",step=it_step)
    

FullPlotCMHD("equilibrium_frozenfield",1.0,0.0,0.0,0.1)
FullPlotCMHD("equilibrium_frozenfield",1.0,0.0,0.0,0.4)
FullPlotCMHD("equilibrium_frozenfield",0.5,0.0,0.0,0.1)
FullPlotCMHD("equilibrium_frozenfield",1.0,0.0,0.0,0.05)
