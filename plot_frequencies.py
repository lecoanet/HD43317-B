import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse      as sparse
from   scipy.linalg      import eig
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.05, 0.3, 0.4, 0.05)
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

i_mu = np.argmin(np.abs(r - (r_inner+r_outer)/2))
r_mu = r[i_mu]

# Magnetic field at the composition gradient
Br = 4.68e5*(r_mu/r)**3 # G

N = np.sqrt(N2)/(2*np.pi)*60*60*24
S = S1/(2*np.pi)*60*60*24
fB = np.sqrt(Br/np.sqrt(np.pi*rho)*np.sqrt(N2)*np.sqrt(2)/r)/(2*np.pi)*60*60*24

f_obs = 0.69162
Prot = 0.897673
f_rot = 1/Prot
m = -1
f_cor = f_obs - m*f_rot
i_inner = np.argmax(N[:i_mid] > f_cor)
i_outer = np.argmin(np.abs(S - f_cor))

plot_axes.plot(r[i_inner:i_outer]/R, f_cor + 0*r[i_inner:i_outer], color='ForestGreen', linestyle='--', label=r'$f_g$', linewidth=2)
plot_axes.plot(r/R, N, label=r'$N$', color='MidnightBlue', linewidth=2)
plot_axes.plot(r/R, S, label=r'$S_1$', color='FireBrick', linewidth=2)
plot_axes.plot(r/R, fB, label=r'$f_B$', color='DarkGoldenrod', linewidth=2)
plt.xlabel(r'$r/R_\star$')
plt.ylabel(r'$f\, ({\rm d}^{-1})$')
plt.yscale('log')
plt.xlim([3e10/R,1.02])
#plt.xlim([r[i_inner-10],r[i_outer+10]])
plt.ylim([0.2,1.5e2])

handles, labels = plot_axes.get_legend_handles_labels()
order = [1,2,3,0]
lg = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', ncol=2)
lg.draw_frame(False)

plt.savefig('frequencies.eps')

