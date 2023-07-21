import numpy as np
import matplotlib.pyplot as plt
from interface import config

from lib.tests import pad_delay_wrapper, mimo_pad_wrapper, mimo_convolve_wrapper
from lib.directions import calculate_coefficients, calculate_delays

whole_samples, adaptive_array = calculate_coefficients()

whole_samples.shape

start_time = 0
end_time = 1
sample_rate = config.fs
time = np.arange(start_time, end_time, 1/sample_rate)
theta = 0
frequency = 8000
amplitude = 1
sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)[:config.N_SAMPLES]


signals = np.repeat(sinewave, config.N_MICROPHONES, axis=0).reshape((config.N_SAMPLES, config.N_MICROPHONES)).T


def create_source(delay):
    pass

print(signals.shape)
out = np.zeros((config.N_MICROPHONES, config.N_SAMPLES), dtype = np.float32)


samp_delay = calculate_delays()
print(samp_delay.shape)


for i in range(samp_delay.shape[-1]):
    delay = samp_delay[3, 3, i]
    print(delay)
    out[i] = amplitude * np.sin(2 * np.pi * frequency * time + delay)[:config.N_SAMPLES]

signals += out
signals = np.float32(signals)
signals /= 2

print(signals.shape)

# plt.plot(signals[0])

a = mimo_pad_wrapper(signals)
# b = mimo_convolve_wrapper(signals)

# print()

cm = plt.cm.get_cmap('jet')

# plt.title(f"Simulated MIMO with {config.N_MICROPHONES} microphones and sinewave {frequency}Hz")
# sc = plt.imshow(a, extent=[-5, 5, -5, 5], cmap=cm, label="test")
# plt.xlabel("Y resolution")
# plt.ylabel(f"X resolution")
# # plt.xlim(-10, 10)

# plt.colorbar(sc)

# plt.show()

def spherical_to_cartesian(theta, phi, rcs):
    x = rcs * np.cos(phi) * np.sin(theta)
    y = rcs * np.sin(phi) * np.sin(theta)
    z = rcs * np.cos(theta)
    return x, y, z




from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
n = len(a)
x = np.linspace(-90, 90, len(a)) * np.pi/180
y = np.linspace(-90, 90, len(a)) * np.pi/180

# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# x, y = np.meshgrid(x, y)
z = a


items = []

for xi, _x in enumerate(x):
    for yi, _y in enumerate(y):
        items.append(spherical_to_cartesian(_x, _y, a[xi, yi]))
k = np.array(items)
print(k.shape)

# plt.scatter(k[:,0], k[:,1], k[:,2])
# plt.show()

# exit()
# sc = ax1.plot_surface(x,y,z, cmap=cm)

# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# ax1.set_zlabel('z axis')
# ax1.set_zlim(0,1)
# # ax1.colorbar(sc)

ax = fig.add_subplot(1,2,1, projection='3d')

 

# ax.grid(False)

# ax.axis('off')

# ax.set_xticks([])

# ax.set_yticks([])

# ax.set_zticks([])

 

# ax.plot_surface(

#     x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),

#     linewidth=0, antialiased=False, alpha=0.5, zorder = 0.5)

 

ax.view_init(azim=300, elev = 30)


phi ,theta = np.linspace(0, 1 * np.pi, n), np.linspace(0, np.pi, n)

PHI, THETA  = np.meshgrid(phi,theta)

R = 1

X = R * np.sin(THETA) * np.cos(PHI)

Y = R * np.sin(THETA) * np.sin(PHI)

# Z = R * np.cos(THETA)

Z = z #* np.cos(THETA)

 
x, y = np.meshgrid(x, y)
ax.set_title("Pad")
ax.plot_surface(

    x, y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),

    linewidth=0, antialiased=False, alpha=1, zorder = 0.5)

# ax = fig.add_subplot(1, 2, 2, projection='3d')

# ax.set_title("Convolve")
# ax.plot_surface(

#     x, y, b, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), label="Convolve",

#     linewidth=0, antialiased=False, alpha=1, zorder = 0.5)

 
 

# ax.plot_wireframe(X, Y, Z, linewidth=0.5, rstride=3, cstride=3)

plt.show()

exit()

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from matplotlib import cm, colors

def interp_array(N1):  # add interpolated rows and columns to array
    N2 = np.empty([int(N1.shape[0]), int(2*N1.shape[1] - 1)])  # insert interpolated columns
    N2[:, 0] = N1[:, 0]  # original column
    for k in range(N1.shape[1] - 1):  # loop through columns
        N2[:, 2*k+1] = np.mean(N1[:, [k, k + 1]], axis=1)  # interpolated column
        N2[:, 2*k+2] = N1[:, k+1]  # original column
    N3 = np.empty([int(2*N2.shape[0]-1), int(N2.shape[1])])  # insert interpolated columns
    N3[0] = N2[0]  # original row
    for k in range(N2.shape[0] - 1):  # loop through rows
        N3[2*k+1] = np.mean(N2[[k, k + 1]], axis=0)  # interpolated row
        N3[2*k+2] = N2[k+1]  # original row
    return N3


vals_theta = np.linspace(-90,90,n)
vals_phi = np.linspace(-90,90,n)

vals_phi, vals_theta = np.meshgrid(vals_phi, vals_theta)

THETA = np.deg2rad(vals_theta)
PHI = np.deg2rad(vals_phi)

# simulate the power data
R = abs(np.cos(PHI)*np.sin(THETA))  # 2 lobes (front and back)

interp_factor = 3  # 0 = no interpolation, 1 = 2x the points, 2 = 4x the points, 3 = 8x, etc

X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
# Z = R * np.cos(THETA)
Z = z

for counter in range(interp_factor):  # Interpolate between points to increase number of faces
    X = interp_array(X)
    Y = interp_array(Y)
    Z = interp_array(Z)

fig = plt.figure()

ax = fig.add_subplot(1,1,1, projection='3d')
ax.grid(True)
ax.axis('on')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

N = np.sqrt(X**2 + Y**2 + Z**2)
Rmax = np.max(N)
N = N/Rmax

axes_length = 1.5
ax.plot([0, axes_length*Rmax], [0, 0], [0, 0], linewidth=2, color='red')
ax.plot([0, 0], [0, axes_length*Rmax], [0, 0], linewidth=2, color='green')
ax.plot([0, 0], [0, 0], [0, axes_length*Rmax], linewidth=2, color='blue')

# Find middle points between values for face colours
N = interp_array(N)[1::2,1::2]

mycol = cm.jet(N)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=mycol, linewidth=0.5, antialiased=True, shade=False)  # , alpha=0.5, zorder = 0.5)

ax.set_xlim([-axes_length*Rmax, axes_length*Rmax])
ax.set_ylim([-axes_length*Rmax, axes_length*Rmax])
ax.set_zlim([-axes_length*Rmax, axes_length*Rmax])

m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(R)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig.colorbar(m, shrink=0.8)
ax.view_init(azim=300, elev=30)

plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as Axes3D
# from numpy import sin,cos,pi, exp,log
# # from tqdm import tqdm
# # import mpl_toolkits.mplot3d.axes3d as Axes3D
# from matplotlib import cm, colors

# # filepath = 'Data_From_CST.txt'

# # vals_theta = []
# # vals_phi = []
# # vals_r = []
# # with open(filepath) as f:
# #   for s in f.readlines()[2:]:
# #     vals_theta.append(float(s.strip().split()[0]))
# #     vals_phi.append(float(s.strip().split()[1]))
# #     vals_r.append(float(s.strip().split()[2]))

# vals_theta = np.linspace(-45, 45, 144)
# vals_phi = np.linspace(-45, 45, 144)
# vals_r = z.flatten()


# theta1d = vals_theta
# theta = np.array(theta1d)/180*pi;

# phi1d = vals_phi
# phi = np.array(phi1d)/180*pi;

# power1d = vals_r
# power = np.array(power1d);
# # power = power-min(power)
# power = 10**(power/10) # I used linscale

# X = power*sin(phi)*sin(theta)
# Y = power*cos(phi)*sin(theta)
# Z = power*cos(theta)

# X = X.reshape([360,181])
# Y = Y.reshape([360,181])
# Z = Z.reshape([360,181])

# def interp_array(N1):  # add interpolated rows and columns to array
#     N2 = np.empty([int(N1.shape[0]), int(2*N1.shape[1] - 1)])  # insert interpolated columns
#     N2[:, 0] = N1[:, 0]  # original column
#     for k in range(N1.shape[1] - 1):  # loop through columns
#         N2[:, 2*k+1] = np.mean(N1[:, [k, k + 1]], axis=1)  # interpolated column
#         N2[:, 2*k+2] = N1[:, k+1]  # original column
#     N3 = np.empty([int(2*N2.shape[0]-1), int(N2.shape[1])])  # insert interpolated columns
#     N3[0] = N2[0]  # original row
#     for k in range(N2.shape[0] - 1):  # loop through rows
#         N3[2*k+1] = np.mean(N2[[k, k + 1]], axis=0)  # interpolated row
#         N3[2*k+2] = N2[k+1]  # original row
#     return N3

# interp_factor=1

# for counter in range(interp_factor):  # Interpolate between points to increase number of faces
#     X = interp_array(X)
#     Y = interp_array(Y)
#     Z = interp_array(Z)

# N = np.sqrt(X**2 + Y**2 + Z**2)
# Rmax = np.max(N)
# N = N/Rmax

# fig = plt.figure(figsize=(10,8))

# ax = fig.add_subplot(1,1,1, projection='3d')
# axes_length = 0.65
# ax.plot([0, axes_length*Rmax], [0, 0], [0, 0], linewidth=2, color='red')
# ax.plot([0, 0], [0, axes_length*Rmax], [0, 0], linewidth=2, color='green')
# ax.plot([0, 0], [0, 0], [0, axes_length*Rmax], linewidth=2, color='blue')

# # Find middle points between values for face colours
# N = interp_array(N)[1::2,1::2]

# mycol = cm.jet(N)

# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=mycol, linewidth=0.5, antialiased=True, shade=False)  # , alpha=0.5, zorder = 0.5)

# ax.set_xlim([-axes_length*Rmax, axes_length*Rmax])
# ax.set_ylim([-axes_length*Rmax, axes_length*Rmax])
# ax.set_zlim([-axes_length*Rmax, axes_length*Rmax])

# m = cm.ScalarMappable(cmap=cm.jet)
# m.set_array(power)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# fig.colorbar(m, shrink=0.8)
# ax.view_init(azim=300, elev=30)

# plt.show()