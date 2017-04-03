# coding=utf-8

from fenics import *

from dolfin.cpp.io import File
from dolfin.cpp.mesh import UnitSquareMesh


def dirichlet_boundary(V, u_D):
    return DirichletBC(V, u_D, lambda x, on_boundary: on_boundary)


def functional_space(mesh):
    return FunctionSpace(mesh, 'P', 2)


if __name__ == '__main__':
    island = UnitSquareMesh(8, 8)

    V = functional_space(island)

    u_D = Expression('1 + x[0] * x[0] + 2 * x[1] * x[1]', degree=2)
    bc = dirichlet_boundary(V, u_D)
    f = Constant(-6.0)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u), grad(v)) * dX
    L = f * v * dX

    u = Function(V)
    solve(a == L, u, bc)

    vtkFile = File('output/poisson/solution.pvd')
    vtkFile << u

    # Проверяем, что решение вышло точным
    error_L2 = errornorm(u_D, u, 'L2')

    vertex_values_u_d = u_D.compute_vertex_values(island)
    vertex_values_u = u.compute_vertex_values(island)

    import numpy as np

    error_max = np.max(np.abs(vertex_values_u - vertex_values_u_d))

    print('error_L2 = ', error_L2)
    print('error_max = ', error_max)
