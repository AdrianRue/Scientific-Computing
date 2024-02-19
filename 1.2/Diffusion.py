import numpy as np
import matplotlib.pyplot as plt
import math

def Calc_Value(Copy_Lattice, i, j, D, dx, dt):


    # As problem can be viewed as symmetrical, values of x on either side as the same
    Value = Copy_Lattice[i,j] + (((D*dt)/(dx**2)) * (Copy_Lattice[i,j+1] + Copy_Lattice[i,j-1] - (2*Copy_Lattice[i,j])))
    
    return Value



def Diffusion(dx, dt, D):
    
    N = int(round(1 / dx))


    Lattice = np.zeros((N,N))

    for i in range(N):

        Lattice[i, N-1] = 1


    for t in np.arange(0, 1 + dt, dt):

        Copy_Lattice = Lattice

        for i in range(N):

            for j in range(1, N - 1):

                Lattice[i,j] = Calc_Value(Copy_Lattice, i, j, D, dx, dt)

        
        #for y in np.arange(0.0, 1 + dx, dx):
        
            #Analytical = (math.erfc((1 - y + 2j) / (2 * math.sqrt(D * t)))) - (math.erf((1 + y + 2j) / (2 * math.sqrt(D * t))))


        if t == 0.001 or t == 0.01 or t == 0.1 or t == 1:
            values_y = []
            values_x = np.arange(0, 1, dx)
            for j in range(N):
                values_y.append(Lattice[0,j])

            plt.plot(values_x, values_y, label = f"t={t:.3f}")
    plt.xlabel('y')
    plt.ylabel('c')
    plt.legend()
    plt.show()



dx = 0.05
dt = 0.0001
D = 1

Diffusion(dx, dt, D)