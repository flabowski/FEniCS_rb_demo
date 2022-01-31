#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:04:11 2021

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from dolfin import (UnitSquareMesh, FunctionSpace, DOLFIN_EPS, Constant, pi,
                    DirichletBC, Function, TrialFunction, TestFunction,
                    Expression, File, plot, project, interpolate)
from dolfin import (UnitIntervalMesh, VectorFunctionSpace, SpatialCoordinate,
                    TrialFunction, TestFunction, sin, pi, dx, inner, grad, ds,
                    Function, solve, dot, assemble)
from ROM.snapshot_manager import Data
# from low_rank_model_construction.basis_function_interpolation import interpolateV, RightSingularValueInterpolator
import matplotlib
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close("all")
cmap = matplotlib.cm.get_cmap('viridis')
plot_width = 12

m = 101  # number of nodes
n = 13  # number of snapshots, e.g. 13 or 25 for a nice spacing
r = 15  # min(m, n)  # reduced rank


dim = min(m, n, r)
mus = 10.**(-np.linspace(0, 12 , n))
n = len(mus)
X = np.empty((m, n))
# x = np.linspace(0, 1, m)

# Create mesh and define function space
mesh = UnitIntervalMesh(m - 1)
V = FunctionSpace(mesh, "Lagrange", 1)


# Define boundary condition
def left(x):
    return x[0] < DOLFIN_EPS


def right(x):
    return x[0] > (1.0 - DOLFIN_EPS)


bcs = [DirichletBC(V, "0", left),
       DirichletBC(V, "0", right)]

# Define variational problem
ubar = TrialFunction(V)
vbar = TestFunction(V)
f = Constant(0.0)  # Expression("0", degree=1)
mu = Constant(0.0)
# a = dot(u / dt, v) * dx + mu * inner(grad(u), grad(v)) * dx
# L = f * v * dx

h = Expression("x[0]", degree=1)
a = dot(ubar, vbar) * dx + mu * inner(grad(ubar), grad(vbar)) * dx
L = -h * vbar * dx

for j, _mu_ in enumerate(mus):
    mu.assign(_mu_)
    u_ = Function(V)
    solve(a == L, u_, bcs)
    X[:, j] = u_.vector().vec().array.ravel()  # u_.compute_vertex_values(mesh)  # u.vector().vec().array.ravel()
x = V.tabulate_dof_coordinates().ravel()

# decompose data
U, S, VT = np.linalg.svd(X, full_matrices=False)

# plot data
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(2*plot_width/2.54, plot_width/2.54))
for j, mu_f in enumerate(mus):
    c = cmap(j/(len(mus)-1))
    ax1.plot(x, X[:, j], marker=".", color=c, label="mu={:.1e}".format(mu_f))
    ax2.plot(VT[:, j], marker=".", color=c, label="mu={:.1e}".format(mu_f))
ax1.set_title("physical space")
ax2.set_title("reduced space")
ax1.legend()
ax2.legend()
ax1.set_xlim([0, 1])
ax1.set_xlabel("x")
ax2.set_xlabel("#")
ax1.set_ylabel("u(x)")
ax2.set_ylabel("VT(#)")
plt.show()

# basis vectors
for j in range(10):
    fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
    ax.plot(U[:, j])
    ax.set_xlabel("x")
    ax.set_ylabel("U(x)")
    plt.title("basis vector {:.0f}".format(j))
    plt.show()

# singular values
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(plot_width/2.54*2, plot_width/2.54))
ax1.plot(np.arange(0, len(S)), S, marker=".")
ax2.plot(np.arange(0, len(S)), np.cumsum(S)/S.sum()*100, marker=".")
ax1.set_xlabel("rank r")
ax1.set_ylabel("singular values")
ax2.set_xlabel("rank r")
ax1.set_yscale('log')
ax2.set_ylabel("Cumulative Energy [%]")
ax1.grid(which="both")
ax2.grid(which="both")
plt.suptitle("dacay of singular values")
plt.tight_layout()
plt.show()


# TODO: redefine U
# ubar = u-h with h = x

R = VectorFunctionSpace(mesh, "R", 0, dim=dim)
# Define reduced basis
rb = [Function(V) for i in range(dim)]
for i in range(dim):
    # rb[i].vector()[:] = U[:, i]  # m
    # rb[i].vector().get_local()
    rb[i].vector().vec().array[:] = U[:, i].copy()
    # rb[i].vector().get_local()

udofs = TrialFunction(R)
vdofs = TestFunction(R)
ubar = sum([udofs[i]*rb[i] for i in range(dim)])
vbar = sum([vdofs[i]*rb[i] for i in range(dim)])

mu.assign(.01)
h = Expression("x[0]", degree=1)

a = dot(ubar, vbar) * dx + mu * inner(grad(ubar), grad(vbar)) * dx
L = -h * vbar * dx

xdofs = Function(R)
solve(a == L, xdofs)  # there are no BCs in R

# # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
# linear_solver, preconditioner = "gmres", "ilu"
# linear_solver, preconditioner = "cg", "hypre_amg"
# linear_solver, preconditioner = 'petsc', 'default'
# linear_solver, preconditioner = 'lu', 'default'
# A = assemble(a)
# b = assemble(L)
# solve(A, xdofs.vector(), b, linear_solver, preconditioner)
# # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

# Get full solution
xx = sum([xdofs[i]*rb[i] for i in range(dim)])
sol = project(xx, V)
sol_np = sol.vector().vec().array
A = assemble(a)

# l, v = np.linalg.eig(A.array())
# fig, ax = plt.subplots()
# plt.plot(l, "ro")

fig, ax = plt.subplots()
plot(sol)  # plot with fenics
plt.plot(x, sol_np, "r.", label="ROM solution")  # plot manually
# mu = 0.01 corresponds to snapshot 4, if there are 25 snapshots
if n == 25:
    ax.plot(x, X[:, 4], "g.", label="FOM solution")
elif n == 13:
    ax.plot(x, X[:, 2], "g.", label="FOM solution")
ax.set_title("solution ubar (u=ubar+x)")
plt.legend()


fig, ax = plt.subplots()
plt.imshow(A.array())
plt.title("A")
plt.show()

fig, ax = plt.subplots()
plt.imshow(U)
