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

data = pickle.load(open('Bcrit.pkl', 'rb'))
m = np.array(data['m'])
ell = np.array(data['ell'])
f = np.array(data['f'])
n_pg = np.array(data['n_pg'])
Bcrit = np.array(data['Bcrit'])

Prot = 0.897673
f_rot = 1/Prot

n_pg_obs_list = [-11, -10, -9, -8, -7, -6, -5, -4, -2, -1]
obs_l1mn1 = 0*ell
notobs_l1mn1 = (ell == 1) * (m == -1)
notobs_l1mn1 = notobs_l1mn1.astype(int)
for n_pg_obs in n_pg_obs_list:
    obs_l1mn1 += (ell == 1) * (m == -1) * ( n_pg == n_pg_obs )
    notobs_l1mn1 -= (ell == 1) * (m == -1) * ( n_pg == n_pg_obs )

obs_l1mn1 = obs_l1mn1 == 1
notobs_l1mn1 = notobs_l1mn1 == 1

n_pg_obs_list = [-15, -11, -10, -9, -6]
obs_l2mn1 = 0*ell
notobs_l2mn1 = (ell == 2) * (m == -1)
notobs_l2mn1 = notobs_l2mn1.astype(int)
for n_pg_obs in n_pg_obs_list:
    obs_l2mn1 += (ell == 2) * (m == -1) * ( n_pg == n_pg_obs )
    notobs_l2mn1 -= (ell == 2) * (m == -1) * ( n_pg == n_pg_obs )

obs_l2mn1 = obs_l2mn1 == 1
notobs_l2mn1 = notobs_l2mn1 == 1

obs_l2m2 = (ell == 2) * (m == 2)

plot_axes.scatter(f[obs_l1mn1], Bcrit[obs_l1mn1], marker='x', color='MidnightBlue', label=r"$(\ell,m)= (1,-1)$")
plot_axes.scatter(f[notobs_l1mn1], Bcrit[notobs_l1mn1], linewidth=0.25, marker='x', color='MidnightBlue')
plot_axes.scatter(f[obs_l2mn1], Bcrit[obs_l2mn1], marker='+', color='FireBrick', s=14, label=r"$(\ell,m) = (2,-1)$")
plot_axes.scatter(f[notobs_l2mn1], Bcrit[notobs_l2mn1], linewidth=0.25, marker='+', color='FireBrick', s=14)
plot_axes.scatter(f[obs_l2m2], Bcrit[obs_l2m2], marker='*', color='DarkGoldenrod', label=r"$(\ell,m)=(2, +2)$")

plot_axes.axhline(4.68e5, color='k', linewidth=0.5)

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$f \ ({\rm d}^{-1})$')
plt.ylabel(r'$B_{\rm crit} \ ({\rm G})$')
lg = plt.legend(loc='upper left')
lg.draw_frame(False)

plt.savefig('Bcrit.eps')

