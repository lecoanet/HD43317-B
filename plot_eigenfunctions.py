
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import publication_settings
from scipy.interpolate import interp1d

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.07, 0.17, 0.25, 0.05)
h_plot, w_plot = (1., 1.)

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

data1 = pickle.load(open('eigenfunctions_weak.pkl','rb'))
r1 = data1['r']
th = data1['th']
th = th.ravel()
kr1 = data1['kr']
uph1 = data1['uph']

data2 = pickle.load(open('eigenfunctions_strong.pkl','rb'))
r2 = data2['r']
kr2 = data2['kr']
uph2 = data2['uph']

def mesh(r, th):
    th = th.ravel()
    th_mid = (th[:-1] + th[1:]) / 2
    th_vert = np.concatenate([[np.pi], th_mid, [0]])
    r = r.ravel()
    r_mid = (r[:-1] + r[1:]) / 2
    r_vert = np.concatenate((r_mid, [0.5]))
    thm, rm = np.meshgrid(th_vert, r_vert, indexing='ij')
    x = np.sin(thm) * rm
    y = np.cos(thm) * rm
    return x,y

def normalize(uph, th):
    dth = -np.gradient(th)
    th_mid = len(th)//2
    uph /= uph[:,th_mid][:,None]
    uph_norm = np.sum(np.sin(th)*np.abs(uph)**2*dth, axis=1)
    uph = uph/np.sqrt(uph_norm)[:, None]
    return uph

def normalize_r(uph, th):
    dth = -np.gradient(th)
    th_mid = len(th)//2
    uph /= uph[th_mid]
    uph_norm = np.sum(np.sin(th)*np.abs(uph)**2*dth)
    uph = uph/np.sqrt(uph_norm)
    return uph

def construct_uph(uph, r, krs, inward=True):

    kr_list = []
    for kr in krs:
        kr_list.append(np.min(kr))
    kr = np.array(kr_list)

    dr = np.gradient(r)
    phase = np.cumsum(kr*dr)
    if inward:
        phase -= phase[-1] - np.pi/2
    else:
        phase += phase1[0]

    uph = normalize(uph, th)
    uph = np.real( np.exp(1j*phase[:, None]) * uph )
    return uph, phase

# inward propagating waves
uph_in1, phase1 = construct_uph(uph1, r1, kr1)
uph_in2, phase2 = construct_uph(uph2, r2, kr2)

r1_fine = np.linspace(np.min(r1), np.max(r1), num=500, endpoint=True)
uph_in1_interp = interp1d(r1, uph_in1, axis=0)
uph_in1 = uph_in1_interp(r1_fine)
x1, y1 = mesh(r1_fine, th)

r2_fine = np.linspace(np.min(r2), np.max(r2), num=500, endpoint=True)
uph_in2_interp = interp1d(r2, uph_in2, axis=0)
uph_in2 = uph_in2_interp(r2_fine)
x2, y2 = mesh(r2_fine, th)

# outward propagating waves
uph_out1, phase1 = construct_uph(uph1, r1, -kr1, inward=False)
uph_out1_interp = interp1d(r1, uph_out1, axis=0)
uph_out1 = uph_out1_interp(r1_fine)

data_res = pickle.load(open('eigenfunctions_resistive.pkl','rb'))
kr_res = -data_res['kr']
th_res = data_res['theta']
uph_res = data_res['uph']
r_res = np.linspace(np.min(r2), np.max(r2), num=500, endpoint=True)
phase_res = kr_res*(r_res - r_res[0])
uph_res = normalize_r(uph_res, th_res)
uph_res = np.real( np.exp(1j*phase_res[:, None]) * uph_res )
x_res, y_res = mesh(r_res, th_res)

mag = np.max(np.abs(uph_in1))*1.2
print(x1.shape[0]//2)
th_mid = x1.shape[0]//2
plot_axes.pcolormesh(-x1[th_mid:], y1[th_mid:], uph_in1[:, th_mid:].T, cmap='RdBu', vmin=-mag, vmax=mag)
plot_axes.pcolormesh( x2[th_mid:], y2[th_mid:], uph_in2[:, th_mid:].T, cmap='RdBu', vmin=-mag, vmax=mag)
plot_axes.pcolormesh(-x1[:th_mid], y1[:th_mid], uph_out1[:, :th_mid].T, cmap='RdBu', vmin=-mag, vmax=mag)
th_mid = x_res.shape[0]//2
plot_axes.pcolormesh( x_res[:th_mid], y_res[:th_mid], uph_res[:, :th_mid].T, cmap='RdBu', vmin=-mag, vmax=mag)
#plot_axes.pcolormesh(x2, y2, uph2.T, cmap='RdBu', vmin=-mag, vmax=mag)
#plot_axes.pcolormesh(-x_r, y_r, uph_r1.T, cmap='RdBu', vmin=-mag, vmax=mag)

th_full = np.linspace(0,2*np.pi,num=300,endpoint=True)
plot_axes.plot(r1[0]*np.sin(th_full), r1[0]*np.cos(th_full), color='k', linewidth=0.5)
#plot_axes.plot(r_inner*np.sin(th_full), r_inner*np.cos(th_full), color='k', linewidth=0.5)
plot_axes.plot(r1[-1]*np.sin(th_full), r1[-1]*np.cos(th_full), color='k', linewidth=0.5)
plot_axes.plot([-0.5, -r1[0]], [0, 0], color='k', linewidth=0.5)
plot_axes.plot([0.5, r1[0]], [0, 0], color='k', linewidth=0.5)
plot_axes.plot([0, 0], [-0.5, -r1[0]], color='k', linewidth=0.5)
plot_axes.plot([0, 0], [0.5, r1[0]], color='k', linewidth=0.5)

plot_axes.annotate(' ', xy=(-0.2,0.2), xytext=(-0.4,0.4), xycoords='data', arrowprops={'arrowstyle': '->'}, va='center')
plot_axes.annotate(' ', xy=(-0.4,-0.4), xytext=(-0.2,-0.2), xycoords='data', arrowprops={'arrowstyle': '->'}, va='center')
plot_axes.annotate(' ', xy=(0.2,0.2), xytext=(0.4,0.4), xycoords='data', arrowprops={'arrowstyle': '->'}, va='center')
plot_axes.annotate(' ', xy=(0.4,-0.4), xytext=(0.2,-0.2), xycoords='data', arrowprops={'arrowstyle': '->'}, va='center')

plot_axes.text(0.1,0.9,'IGW',va='center',ha='center',fontsize=10,transform=plot_axes.transAxes)
plot_axes.text(0.1,0.07,'IGW',va='center',ha='center',fontsize=10,transform=plot_axes.transAxes)
plot_axes.text(0.9,0.07,'AW',va='center',ha='center',fontsize=10,transform=plot_axes.transAxes)
plot_axes.text(0.9,0.9,'IGW',va='center',ha='center',fontsize=10,transform=plot_axes.transAxes)

plot_axes.text(0.01,0.97,r'$n_{\rm pg}=-15$',va='center',ha='left',fontsize=10,transform=plot_axes.transAxes)
plot_axes.text(0.99,0.97,r'$n_{\rm pg}=-16$',va='center',ha='right',fontsize=10,transform=plot_axes.transAxes)

plot_axes.set_xticks([-0.5, 0, 0.5])
plot_axes.set_yticks([-0.5, 0, 0.5])
plot_axes.set_xlabel(r'$x/R_\star$')
plot_axes.set_ylabel(r'$z/R_\star$')
plot_axes.set_title(r'$u_\phi$',fontsize=12)
plt.savefig('eigenfunctions.png', dpi=600)

