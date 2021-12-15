import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
from tomso import gyre
import pickle
import glob

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

Prot = 0.897673
f_rot = 1/Prot
om_rot = 2*np.pi*f_rot/24/60/60

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

def calc_Bc(f_obs, m, ell, N2_0, R, Br_list): 

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
        
#        for i in range(len(Br_list)):
#            print(i)
#            print(Br_list[i])
#            print(kr_list[i])
            
        for i in range(len(Br_list)):
            if len(kr_list[i]) < ell:
                break
        print(i)
        Br_crit = Br0_list[i+1]
    else:
        Br_crit = None

    return Br_crit


files = glob.glob('models/*.data.GYRE')

Bc_15_list = []
Bc_16_list = []

for file in files:
    stem = file[7:-10]

    data = np.loadtxt(file)
    r = data[:,1]
    rho = data[:,6]
    N2 = data[:,8] # (rad/s)^2
    R = r[-1]

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
    logger.info(N2_0)

    Br0_list = np.linspace(4e5, 5e5, num=500, endpoint=False)
    Br_list = Br0_list*(r_spike/r)**3/np.sqrt(4*np.pi*rho) # radial alfven velocity

    gyre_file = 'GYRE/'+stem+'_l2m-1_frequencies.ad'
    logger.info(gyre_file)
    mode_data = gyre.load_summary(gyre_file)

    f_obs = mode_data['Refreq'][-14]
    m = mode_data['m'][-14]
    ell = mode_data['l'][-14]
    n_pg = mode_data['n_pg'][-14]

    logger.info(n_pg)

    Bcrit = calc_Bc(f_obs, m, ell, N2_0, R, Br_list)
    Bc_15_list.append(Bcrit)
    logger.info(Bcrit)

    f_obs = mode_data['Refreq'][-15]
    m = mode_data['m'][-15]
    ell = mode_data['l'][-15]
    n_pg = mode_data['n_pg'][-15]

    logger.info(n_pg)

    Bcrit = calc_Bc(f_obs, m, ell, N2_0, R, Br_list)
    Bc_16_list.append(Bcrit)
    logger.info(Bcrit)

if rank == 0:
    data = {'Bc_15': Bc_15_list, 'Bc_16': Bc_16_list}
    pickle.dump(data, open('Bcrit_models.pkl', "wb"))

