import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
from tomso import gyre
import pickle

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

ell = 2
if ell == 1:
    file = 'GYRE/m5800_z014_ov004_profile_at_xc540_l1m-1_frequencies.ad'
    mode_data = gyre.load_summary(file)

    f_obs = mode_data['Refreq'][-10]
    logger.info(f_obs)
    name = 'wavenumbers_ell1.pkl'
else:
    file = 'GYRE/m5800_z014_ov004_profile_at_xc540_l2m-1_frequencies.ad'
    mode_data = gyre.load_summary(file)

    f_obs = mode_data['Refreq'][-14]
    logger.info(f_obs)
    name = 'wavenumbers_ell2.pkl'

data = np.loadtxt('best.data.GYRE')
r = data[:,1]
rho = data[:,6]
N2 = data[:,8] # (rad/s)^2
R = r[-1]
Br = 1312*(R/r)**3 # G

i_mid = np.argmin(np.abs(r-R/2))
i_inner = np.where(N2[:i_mid]>2e-6)[0][0]
i_outer = np.where(N2[:i_mid]>2e-6)[0][-1]

r_inner = r[i_inner]
r_outer = r[i_outer]

i_spike = np.argmin(np.abs(r - (r_inner+r_outer)/2))
r_spike = r[i_spike]

i_min = np.argmin(N2[i_spike:i_mid]) + i_spike
i_max = np.argmax(N2[:i_spike])

r_list1 = np.linspace(r[i_max], r[i_min], num=15, endpoint=False)
r_list2 = np.linspace(r[i_min], r[i_mid], num=100, endpoint=True)
r_list = np.concatenate((r_list1, r_list2))
N2_list = np.interp(r_list, r, N2)
rho_list = np.interp(r_list, r, rho)

Bcrit = 4.68e5 # G
#Bcrit = 51767.38227551298
Br_list = Bcrit*(r_spike/r_list)**3/np.sqrt(4*np.pi*rho_list) # radial alfven velocity

# Parameters
Nphi = 4
dtype = np.complex128

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)

#res1 = 192
#res2 = 256
res1 = 128
res2 = 192

basis_lres = d3.SphereBasis(coords, (Nphi, res1), radius=1, dtype=dtype)
basis_hres = d3.SphereBasis(coords, (Nphi, res2), radius=1, dtype=dtype)
phi, theta = basis_hres.local_grids()

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))
C2 = lambda A: d3.MulCosine(d3.MulCosine(A))

def solve(basis, N2, Br, Om, r, m):
    print(N2, Br, Om, r)
    zcross = lambda A: d3.MulCosine(d3.skew(A))
    C2 = lambda A: d3.MulCosine(d3.MulCosine(A))
    kr2 = dist.Field(name='kr2')

    u = dist.VectorField(coords, name='u', bases=basis)
    ur = dist.Field(name='ur', bases=basis)
    p = dist.Field(name='p', bases=basis)

    problem = d3.EVP([ur, u, p], eigenvalue=kr2, namespace=locals())
    problem.add_equation("N2*ur + p = 0")
    problem.add_equation("u + 1j*2*Om*zcross(u) + 1j*grad(p)/r - kr2*Br**2*C2(u) = 0")
    problem.add_equation("div(u)/r + 1j*kr2*ur = 0")

    # Solve
    solver = problem.build_solver()
    for sp in solver.subproblems:
        if sp.group[0] == m:
            solver.solve_dense(sp)

    vals = solver.eigenvalues
    vecs = solver.eigenvectors

    bad = (np.abs(vals) > 1e9)
    vals[bad] = np.nan
    vecs = vecs[:, np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]

    vecs = vecs[:, np.abs(np.imag(vals)) < 10]
    vals = vals[np.abs(np.imag(vals)) < 10]
    vecs = vecs[:, vals.real > 0]
    vals = vals[vals.real > 0]

    i = np.argsort(np.sqrt(vals).real)
    solver.eigenvalues, solver.eigenvectors = vals[i], vecs[:, i]

    return vals

def converged_vals(i, Om, R, m, Br=None, N2=None, r=None):

    if Br is None:
        Br = Br_list[i]/(R*om_cor)
        N2 = N2_list[i]
        r = r_list[i]/R

    vals1 = solve(basis_lres, N2, Br, Om, r, m)
    vals2 = solve(basis_hres, N2, Br, Om, r, m)

    vals_discrete = []
    vals_continuum = []
    for val in vals2:
        if np.min(np.abs(val - vals1))/np.abs(val) < 1e-7:
            vals_discrete.append(val)
        else:
            vals_continuum.append(val)
    vals_discrete = np.array(vals_discrete)
    vals_continuum = np.array(vals_continuum)

    return vals_discrete, vals_continuum

Prot = 0.897673
f_rot = 1/Prot
m = -1
f_cor = f_obs - m*f_rot
#f_cor = f_obs
om_rot = 2*np.pi*f_rot/24/60/60
om_cor = 2*np.pi*f_cor/24/60/60
Om = om_rot/om_cor
N2_list = N2_list/om_cor**2

kr_IGW = []
kr_AW = []
r_local = []
for i in range(rank, len(r_list), size):
    r_local.append(r_list[i]/R)

    vals_discrete, vals_continuum = converged_vals(i, Om, R, m)
    kr = np.sqrt(vals_discrete)
    order = np.argsort(kr.real)

    kr_IGW.append( kr[order][ell-1] )
    kr_AW.append( np.min(np.sqrt(vals_continuum)) )
   
    logger.info(kr_IGW[-1])
    logger.info(kr_AW[-1])

kr_IGW_list = MPI.COMM_WORLD.gather(kr_IGW, root=0)
kr_AW_list = MPI.COMM_WORLD.gather(kr_AW, root=0)
r_list_list = MPI.COMM_WORLD.gather(r_local, root=0)

if rank == 0:
    IGW_list = []
    AW_list = []
    r_list = []
    
    for r_l, kr_IGW, kr_AW in zip(r_list_list, kr_IGW_list, kr_AW_list):
        for r, kr1, kr2 in zip(r_l, kr_IGW, kr_AW):
            IGW_list.append(kr1)
            AW_list.append(kr2)
            r_list.append(r)

    r_list = np.array(r_list)
    i = np.argsort(r_list)
    r_list = r_list[i]
    IGW_list = np.array(IGW_list)[i]
    AW_list = np.array(AW_list)[i]

    data = {'r': r_list, 'kr_IGW':IGW_list, 'kr_AW':AW_list}
    pickle.dump(data, open( name, "wb" ))

