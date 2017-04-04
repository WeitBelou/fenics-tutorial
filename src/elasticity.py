# coding=utf-8
from dolfin.cpp.io import File
from dolfin.cpp.mesh import Point
from fenics import *
from mshr import generate_mesh, Cylinder

if __name__ == '__main__':
    # Объявляем кучу переменных
    L = 0.2
    R = 1.0
    mu = 1
    rho = 1
    delta = R / L
    gamma = 0.4 * delta ** 2
    beta = 1.25
    lambda_ = beta
    g = gamma

    # Создаём mesh и функциональное пространство
    mesh = generate_mesh(Cylinder(Point(0.0, 0.0, L),
                                  Point(0.0, 0.0, 0.0),
                                  R, R), 64)
    V = VectorFunctionSpace(mesh, 'P', 1)

    # Настраиваем граничные условия
    tol = 1e-4


    def clamped_boundary(x, on_boundary):
        return on_boundary and x[2] < tol


    bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)


    # Определим \eps и \sigma
    def epsilon(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)


    def sigma(u):
        return lambda_ * nabla_div(u) * Identity(3) + 2 * mu * epsilon(u)


    u = TrialFunction(V)
    d = u.geometric_dimension()
    v = TestFunction(V)
    f = Constant((0, 0, -rho * g))
    T = Constant((0, 0, 0))
    a = inner(sigma(u), epsilon(v)) * dX
    L = dot(f, v) * dX + dot(T, v) * ds

    u = Function(V)
    solve(a == L, u, bc)

    vtk_file = File("../output/elasticity/displacement.pvd")
    vtk_file << u

    s = sigma(u) - (1. / 3) * tr(sigma(u)) * Identity(3)
    von_Mises = sqrt(3. / 2 * inner(s, s))
    V = FunctionSpace(mesh, 'P', 1)
    von_Mises = project(von_Mises, V)

    vtk_file = File("../output/elasticity/stress.pvd")
    vtk_file << von_Mises