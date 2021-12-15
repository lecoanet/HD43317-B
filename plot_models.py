import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse      as sparse
from   scipy.linalg      import eig
import publication_settings
import pickle

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

data = pickle.load(open('Bcrit_models.pkl', 'rb'))

Bc_15 = np.array(data['Bc_15'])
Bc_16 = np.array(data['Bc_16'])

bins = np.linspace(4.3e5, 4.9e5, num=30)

Bc = (Bc_16+Bc_15)/2

plot_axes.hist(Bc, bins, color='DarkGoldenrod', edgecolor='k')
#plot_axes.hist(Bc_15, bins, color='FireBrick', alpha=0.5, edgecolor='k')
#plot_axes.hist(Bc_16, bins, color='MidnightBlue', alpha=0.5, edgecolor='k')

height = lambda i: 4.5 + 0.2*i

for i in range(len(Bc_15)):
   plot_axes.plot([Bc_15[i], Bc_16[i]],[height(i), height(i)], color='k')
   if i == 0:
       label = r'$B_{\rm crit, -16}$'
   else:
       label = None
   plot_axes.scatter([Bc_16[i]],[height(i)], marker='x', color='MidnightBlue', zorder=5, label=label)
   if i == 0:
       label = r'$B_{\rm crit, -15}$'
   else:
       label = None
   plot_axes.scatter([Bc_15[i]],[height(i)], marker='x', color='FireBrick', zorder=5, label=label)
   if i == 0:
       label = r'$B_{\rm crit}$'
   else:
       label = None
   plot_axes.scatter([Bc[i]],[height(i)], marker='o', color='DarkGoldenrod', zorder=4, label=label)

plot_axes.axhline(4.25, color='k')

plot_axes.set_yticks([0, 2, 4])

lg = plt.legend(loc='lower right', handletextpad=0.)
lg.draw_frame(False)
plot_axes.set_xlabel(r'$B_{\rm crit}$')
plot_axes.set_ylabel(r'${\rm number}$')
plot_axes.yaxis.set_label_coords(-0.1, 0.25)

Bc_mean = np.mean(Bc)
Bc_rms = np.sqrt(np.mean((Bc - Bc_mean)**2))
print(Bc_mean)
print(Bc_rms)

plt.savefig('models.eps')

