# coding=utf-8
from dolfin import *
from dolfin.cpp.io import File
from dolfin.cpp.mesh import UnitSquareMesh


def main():
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'P', 1)

    u_D = Expression('1 + x[0] + x[1] + 0.3 * t',
                     degree=2, t=0)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    u_n = project(u_D, V)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(2.0)

    dt = 0.3
    F = u * v * dX + dt * dot(grad(u), grad(v)) * dX - (u_n + dt * f) * v * dX
    a, L = lhs(F), rhs(F)

    u = Function(V)
    t = 0

    num_steps = int(30 / dt)

    vtk_file = File('../output/solution.pvd')

    for n in range(num_steps):
        t += dt
        u_D.t = t

        solve(a == L, u, bc)
        u_n.assign(u)

        vtk_file << u


if __name__ == '__main__':
    main()
