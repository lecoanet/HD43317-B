import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from tomso import gyre
import pickle

# Parameters
Nphi = 4
dtype = np.complex128

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)

#Nlres = 127
#Nhres = 191
Nlres = 96
Nhres = 128
basis = d3.SphereBasis(coords, (Nphi, Nlres), radius=1, dtype=dtype)
basis2 = d3.SphereBasis(coords, (Nphi, Nhres), radius=1, dtype=dtype)

# mode frequencies
Prot = 0.897673
f_rot = 1/Prot

file = 'GYRE/m5800_z014_ov004_profile_at_xc540_l2m-1_frequencies.ad'
mode_data = gyre.load_summary(file)
f_obs = mode_data['Refreq'][-15]
logger.info(f_obs)

m = -1
ell = 2
f_cor = f_obs - m*f_rot
#f_cor = f_obs
om_rot = 2*np.pi*f_rot/24/60/60
om_cor = 2*np.pi*f_cor/24/60/60
Om = om_rot/om_cor

# background data
data = np.loadtxt('best.data.GYRE')
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

# Magnetic field at the composition gradient
Br = 4.68e5 # G

N2 = N2_0/om_cor**2
Br = Br*(r_spike/r)**3/np.sqrt(4*np.pi*rho)/(R*om_cor)
r = r/R

def dense_solve(basis, N2, Br, Om, r, eta):
    # Substitutions
    zcross = lambda A: d3.MulCosine(d3.skew(A))
    C2 = lambda A: d3.MulCosine(d3.MulCosine(A))
    C = lambda A: d3.MulCosine(A)

    u = dist.VectorField(coords, name='u', bases=basis)
    b = dist.VectorField(coords, name='b', bases=basis)
    dbdr = dist.VectorField(coords, name='dbdr', bases=basis)
    ur = dist.Field(name='ur', bases=basis)
    p = dist.Field(name='p', bases=basis)
    kr = dist.Field(name='kr')    

    problem = d3.EVP([ur, u, p, b, dbdr], eigenvalue=kr, namespace=locals())
    problem.add_equation("N2*ur + kr*p = 0")
    problem.add_equation("-1j*u + 2*Om*zcross(u) + grad(p)/r - 1j*Br*kr*C(b) = 0")
    problem.add_equation("div(u)/r + 1j*kr*ur = 0")
    problem.add_equation("-1j*b - 1j*Br*kr*C(u) + eta*kr*dbdr = 0")
    problem.add_equation("dbdr - kr*b = 0")
    solver = problem.build_solver()
    for sp in solver.subproblems:
            if sp.group[0] == m:
                solver.solve_dense(sp)

    vals = solver.eigenvalues
    
    bad = (np.abs(vals) > 1e9)
    vals[bad] = np.nan
    vals = vals[np.isfinite(vals)]
    vals = vals[vals.real > 0]

    # modes of interest have weak decay rates
    vals = vals[np.abs(vals.real) > 2*np.abs(vals.imag) ]
    
    i = np.argsort(vals.real)
    vals = vals[i]
    return vals

def sparse_solve(val, basis, N2, Br, Om, r, eta):
    # Substitutions
    zcross = lambda A: d3.MulCosine(d3.skew(A))
    C2 = lambda A: d3.MulCosine(d3.MulCosine(A))
    C = lambda A: d3.MulCosine(A)

    u = dist.VectorField(coords, name='u', bases=basis)
    b = dist.VectorField(coords, name='b', bases=basis)
    dbdr = dist.VectorField(coords, name='dbdr', bases=basis)
    ur = dist.Field(name='ur', bases=basis)
    p = dist.Field(name='p', bases=basis)
    kr = dist.Field(name='kr')

    problem = d3.EVP([ur, u, p, b, dbdr], eigenvalue=kr, namespace=locals())
    problem.add_equation("N2*ur + kr*p = 0")
    problem.add_equation("-1j*u + 2*Om*zcross(u) + grad(p)/r - 1j*Br*kr*C(b) = 0")
    problem.add_equation("div(u)/r + 1j*kr*ur = 0")
    problem.add_equation("-1j*b - 1j*Br*kr*C(u) + eta*kr*dbdr = 0")
    problem.add_equation("dbdr - kr*b = 0")
    solver = problem.build_solver()
    for sp in solver.subproblems:
            if sp.group[0] == m:
                solver.solve_sparse(sp, 1, val)

    return solver

eta = 1e-6
vals1 = dense_solve(basis, N2, Br, Om, r, eta)
vals2 = dense_solve(basis2, N2, Br, Om, r, eta)

vals = []
for val in vals1:
    if np.min(np.abs(val - vals2))/np.abs(val) < 1e-7:
        vals.append(val)
vals = np.array(vals)
logger.info(vals)

basis = d3.SphereBasis(coords, (Nphi, 256), radius=1, dtype=dtype)
solver = sparse_solve(vals[ell-1], basis, N2, Br, Om, r, 5e-7)
logger.info(solver.eigenvalues[0])
basis = d3.SphereBasis(coords, (Nphi, 1024), radius=1, dtype=dtype)
solver = sparse_solve(solver.eigenvalues[0], basis, N2, Br, Om, r, 1e-7)
logger.info(solver.eigenvalues[0])
basis = d3.SphereBasis(coords, (Nphi, 4096), radius=1, dtype=dtype)
solver = sparse_solve(solver.eigenvalues[0], basis, N2, Br, Om, r, 1e-8)
logger.info(solver.eigenvalues[0])
basis = d3.SphereBasis(coords, (Nphi, 4096), radius=1, dtype=dtype)
solver = sparse_solve(solver.eigenvalues[0], basis, N2, Br, Om, r, 3e-9)
logger.info(solver.eigenvalues[0])
basis = d3.SphereBasis(coords, (Nphi, 4096), radius=1, dtype=dtype)
solver = sparse_solve(solver.eigenvalues[0], basis, N2, Br, Om, r, 1e-9)
logger.info(solver.eigenvalues[0])
basis = d3.SphereBasis(coords, (Nphi, 8192), radius=1, dtype=dtype)
solver = sparse_solve(solver.eigenvalues[0], basis, N2, Br, Om, r, 3e-10)
logger.info(solver.eigenvalues[0])
basis = d3.SphereBasis(coords, (Nphi, 16384), radius=1, dtype=dtype)
solver = sparse_solve(solver.eigenvalues[0], basis, N2, Br, Om, r, 3e-10)
logger.info(solver.eigenvalues[0])
# convergence study
basis_eig = d3.SphereBasis(coords, (Nphi, 32768), radius=1, dtype=dtype)
solver_eig = sparse_solve(solver.eigenvalues[0], basis_eig, N2, Br, Om, r, 3e-10)
logger.info(solver_eig.eigenvalues[0])
basis_eig = d3.SphereBasis(coords, (Nphi, 65536), radius=1, dtype=dtype)
solver_eig = sparse_solve(solver_eig.eigenvalues[0], basis_eig, N2, Br, Om, r, 3e-10)
logger.info(solver_eig.eigenvalues[0])


for sp in solver.subproblems:
    if sp.group[0] == m:
        break
solver.set_state(0, sp.subsystems[0])
uph = solver.state[1]['g'][0,0]
phi, theta = basis.local_grids()
theta = theta.ravel()

data = {'uph': uph, 'theta': theta, 'kr': solver.eigenvalues[0]}
pickle.dump(data, open( "eigenfunctions_resistive.pkl", "wb" ))

