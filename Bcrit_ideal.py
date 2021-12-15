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

weak = True
if weak:
    f_obs = 0.69162
    name = 'wavenumbers_weak.pkl'
else:
    f_obs = 0.6
    name = 'wavenumbers_strong.pkl'

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

r = r[i_max]
N2_0 = N2[i_max]
rho = rho[i_max]

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

def converged_vals(N2, Br, Om, R, m):

    vals1 = solve(basis_lres, N2, Br, Om, r/R, m)
    vals2 = solve(basis_hres, N2, Br, Om, r/R, m)

    vals = []
    for val in vals2:
        if np.min(np.abs(val - vals1))/np.abs(val) < 1e-7:
            vals.append(val)

    return vals

Prot = 0.897673
f_rot = 1/Prot
om_rot = 2*np.pi*f_rot/24/60/60

logB = np.linspace(5.4, 7.2, num=500, endpoint=True)
Br0_list = 10**(logB) # G
Br_list = Br0_list*(r_spike/r)**3/np.sqrt(4*np.pi*rho) # radial alfven velocity

file = 'm5800_z014_ov004_profile_at_xc540_l2m2_frequencies.ad'
mode_data = gyre.load_summary(file)

f_list =  np.array(mode_data['Refreq'][-5])
m_list =  np.array(mode_data['m'][-5])
ell_list =  np.array(mode_data['l'][-5])
n_pg_list =  np.array(mode_data['n_pg'][-5])

num = -20

#file = 'm5800_z014_ov004_profile_at_xc540_l1m1_frequencies.ad'
#mode_data = gyre.load_summary(file)
#
#f_list =  np.concatenate( ((f_list,), np.array(mode_data['Refreq'][num:])) )
#m_list = np.concatenate( ((m_list,), np.array(mode_data['m'][num:])) )
#ell_list = np.concatenate( ((ell_list,), np.array(mode_data['l'][num:])) )
#n_pg_list = np.concatenate( ((n_pg_list,), np.array(mode_data['n_pg'][num:])) )

file = 'm5800_z014_ov004_profile_at_xc540_l1m-1_frequencies.ad'
mode_data = gyre.load_summary(file)

f_list =  np.concatenate( ((f_list,), np.array(mode_data['Refreq'][num:])) )
m_list = np.concatenate( ((m_list,), np.array(mode_data['m'][num:])) )
ell_list = np.concatenate( ((ell_list,), np.array(mode_data['l'][num:])) )
n_pg_list = np.concatenate( ((n_pg_list,), np.array(mode_data['n_pg'][num:])) )

file = 'm5800_z014_ov004_profile_at_xc540_l2m-1_frequencies.ad'
mode_data = gyre.load_summary(file)

f_list =  np.concatenate( (f_list, np.array(mode_data['Refreq'][num:])) )
m_list = np.concatenate( (m_list, np.array(mode_data['m'][num:])) )
ell_list = np.concatenate( (ell_list, np.array(mode_data['l'][num:])) )
n_pg_list = np.concatenate( (n_pg_list, np.array(mode_data['n_pg'][num:])) )

Bc_list = []

for mode_index in range(len(f_list)):

    Br_list = Br0_list*(r_spike/r)**3/np.sqrt(4*np.pi*rho) # radial alfven velocity

    m = m_list[mode_index]
    ell = ell_list[mode_index]
    f_obs = f_list[mode_index]
    f_cor = f_obs - m*f_rot
    om_cor = 2*np.pi*f_cor/24/60/60
    Om = om_rot/om_cor
    N2_norm = N2_0/om_cor**2

    krs = []
    Br_local = []
    for i in range(rank, len(Br_list), size):
    
        Br_local.append( Br0_list[i] )
        Br = Br_list[i]/(R*om_cor)
        vals = converged_vals(N2_norm, Br, Om, R, m)
        logger.info(np.sqrt(vals))
        if len(vals) > 0:
            krs.append( np.sqrt(vals) )
        else:
            krs.append( [] )
       
    krs_list = MPI.COMM_WORLD.gather(krs, root=0)
    Br_list_list = MPI.COMM_WORLD.gather(Br_local, root=0)

    if rank == 0:
        kr_list = []
        Br_list = []
        
        for Br_l, krs in zip(Br_list_list, krs_list):
            for Br, kr in zip(Br_l, krs):
                kr_list.append(kr)
                Br_list.append(Br)

        Br_list = np.array(Br_list)
        i = np.argsort(Br_list)
        Br_list = Br_list[i]
        kr_list = np.array(kr_list)[i]
    
        for i in range(len(Br_list)):
            print(i)
            print(Br_list[i])
            print(kr_list[i])
    
        for i in range(len(Br_list)):
            if len(kr_list[i]) < ell:
                break
        print(i)
        Br_crit = Br0_list[i+1]

        logger.info(f_cor)
        logger.info(Br_crit)

        Bc_list.append( Br_crit )

if rank == 0:

    data = {'m': m_list, 'ell': ell_list, 'f':f_list, 'n_pg':n_pg_list, 'Bcrit': Bc_list}
    pickle.dump(data, open('Bcrit.pkl', "wb"))

