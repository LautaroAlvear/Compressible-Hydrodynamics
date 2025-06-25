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
        step = 10

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
