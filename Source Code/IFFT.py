import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def inverse_fft_function(x0, L, N, v, nu, t):
    """
    Compute the inverse Fourier transform numerically using inverse FFT.

    Parameters:
    - x0: Initial position
    - L: Length of the spatial domain
    - N: Number of grid points
    - v: Transport speed
    - nu: Diffusion coefficient
    - t: Time at which to evaluate the inverse transform

    Returns:
    - u: Reconstructed solution u(x, t)
    """
    dx = (L - x0) / N
    

    # frequency domain
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    omega = k
    u_hat = np.sqrt(np.pi) * np.exp(-(1j * omega * x0 + 1j * v * omega + np.pi ** 2 * omega ** 2 + nu * omega ** 2) * t)

    u_hat = u_hat.reshape(1, -1)

    # Compute inverse 2D FFT
    u = np.fft.ifft2(u_hat).real

    return u[0]




# Parameters
x0 = -3
L = 10  # Spatial domain length
N = 256  # Number of grid points
v = 2  # Transport speed
nu = 0.001  # Diffusion coefficient
Tmax = 20 # max iter




U = []
for j in range(Tmax):
    u = inverse_fft_function(x0, L, N, v, nu, j)
    U.append(u)
    
UU = np.array(U)

tt = np.arange(0,Tmax,1)
x = np.linspace(x0, L, N)

## End-to-End Plotting (x as variable, t as parameter)




# for k in range(Tmax):
#     plt.plot(x,UU[k])
#     plt.show()

fig, ax = plt.subplots(2,2,figsize=(20,20))
fig.suptitle('Inverse FFT Solution')

ax[0,0].plot(x,UU[2],label='0')
ax[0,1].plot(x,UU[5],label='10')
ax[1,0].plot(x,UU[10],label='15')
ax[1,1].plot(x,UU[19],label='19')

ax[0,0].set_title('t = 2')
ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('u')


ax[0,1].set_title('t = 5')
ax[0,1].set_xlabel('x')
ax[0,1].set_ylabel('u')


ax[1,0].set_title('t = 10')
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('u')


ax[1,1].set_title('t = tmax')
ax[1,1].set_xlabel('x')
ax[1,1].set_ylabel('u')

plt.show()

# X, T = np.meshgrid(x,tt)

# # 3D Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, UU, cmap='viridis')
# ax.set_title('Inverse FFT Solution')
# ax.legend()
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u(x, t)')
# ax.grid()
# plt.show()
