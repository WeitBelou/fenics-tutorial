# coding=utf-8
from dolfin.cpp.function import near
from dolfin.cpp.io import File
from dolfin.cpp.mesh import Point
from fenics import *
from mshr import generate_mesh, Cylinder


def create_cylinder_mesh(l, r):
    return generate_mesh(Cylinder(Point(0.0, 0.0, l),
                                  Point(0.0, 0.0, 0.0),
                                  r, r), 64)


if __name__ == '__main__':
    l = 0.2
    r = 1.0
    mesh = create_cylinder_mesh(l, r)

    # Сперва решаем относительно температуры
    V = FunctionSpace(mesh, 'P', 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    upper_bc = DirichletBC(V, Constant(200), lambda x, on_boundary: on_boundary and near(x[2], l))
    default_bc = DirichletBC(V, Constant(270), lambda x, on_boundary: on_boundary and not near(x[2], l))
    f = Constant(0)

    a = dot(u, v) * dx
    L = f * v * dx

    theta = Function(V, name='T')
    solve(a == L, theta, bcs=[upper_bc, default_bc])

    File("../output/thermo_elasticity/temperature.pvd") << theta