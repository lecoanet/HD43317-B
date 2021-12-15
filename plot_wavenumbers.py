import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse      as sparse
from   scipy.linalg      import eig
import pickle
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.05, 0.3, 0.35, 0.08)
h_plot, w_plot = (1., 1./publication_settings.golden_mean)

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plots
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot ) / h_total
width = w_plot / w_total
height = h_plot / h_total
plot_axes = fig.add_axes([left, bottom, width, height])

data = np.loadtxt('best.data.GYRE')
r = data[:,1]
p = data[:,4]
rho = data[:,6]
gamma = data[:,9]
N2 = data[:,8] # (rad/s)^2
R = r[-1]
cs = np.sqrt(gamma*p/rho)
S1 = cs*np.sqrt(2)/r

i_mid = np.argmin(np.abs(r-R/2))
i_inner = np.where(N2[:i_mid]>2e-6)[0][0]
i_outer = np.where(N2[:i_mid]>2e-6)[0][-1]

r_inner = r[i_inner]
r_outer = r[i_outer]

i_spike = np.argmin(np.abs(r - (r_inner+r_outer)/2))
r_spike = r[i_spike]

i_min = np.argmin(N2[i_spike:i_mid]) + i_spike
r_min = r[i_min]

data = pickle.load(open('wavenumbers_ell1.pkl','rb'))
r = data['r']
kr1_IGW = data['kr_IGW']
kr_AW = data['kr_AW']

data = pickle.load(open('wavenumbers_ell2.pkl','rb'))
kr2_IGW = data['kr_IGW']

plot_axes.fill_between(r, kr_AW.real, [1e4]*len(r), facecolor='#a6c9df', label='AW')
plot_axes.plot(r, kr_AW.real, color='#4682b8', linewidth=0.5)
plot_axes.plot(r, kr1_IGW.real, color='FireBrick', linewidth=2, label=r'$\ell=1$ IGW')
plot_axes.plot(r, kr2_IGW.real, color='DarkGoldenrod', linewidth=2, label=r'$\ell=2$ IGW')
plot_axes.axvline(r[0], color='k', linewidth=0.5)
plot_axes.axvline(r_min/R, color='k', linestyle='--', linewidth=0.5)

plt.yscale('log')
plt.xlabel(r'$r/R_\star$')
plt.ylabel(r'$k_r \ R_\star$')
plt.ylim([None, 4e3])
plt.xlim([None,0.5])
lg = plt.legend(loc='center right')
lg.draw_frame(False)

plt.savefig('wavenumbers.eps')

