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
