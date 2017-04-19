# coding=utf-8
from dolfin.cpp.function import near
from dolfin.cpp.io import File
from dolfin.cpp.mesh import Point
from fenics import *
from mshr import generate_mesh, Cylinder


class IceIsland:
    def __init__(self, full_height=10, underwater_height=5, radius=150):
        self.H = full_height
        self.h = underwater_height
        self.R = radius

        self.mesh = self._create_mesh()

    def _create_mesh(self):
        return generate_mesh(Cylinder(Point(0.0, 0.0, 0.0),
                                      Point(0.0, 0.0, self.H),
                                      self.R, self.R), 100)

    def on_bottom_surface(self, x, on_boundary):
        return on_boundary and near(x[2], 0)

    def on_upper_surface(self, x, on_boundary):
        return on_boundary and near(x[2], self.H)

    def on_underwater_side_surface(self, x, on_boundary):
        return on_boundary and 0 < x[2] < self.h

    def on_above_water_side_surface(self, x, on_boundary):
        return on_boundary and self.h < x[2] < self.H


def solve_heat_equation(geo):
    V = FunctionSpace(geo.mesh, 'P', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    bcs = [
        DirichletBC(V, Constant(200), lambda x, on_boundary: geo.on_upper_surface(x, on_boundary)),
        DirichletBC(V, Constant(200), lambda x, on_boundary: geo.on_above_water_side_surface(x, on_boundary)),

        DirichletBC(V, Constant(270), lambda x, on_boundary: geo.on_underwater_side_surface(x, on_boundary)),
        DirichletBC(V, Constant(270), lambda x, on_boundary: geo.on_bottom_surface(x, on_boundary))
    ]

    f = Constant(0.0)
    a = dot(u, v) * dx
    L = f * v * dx

    temperature = Function(V, name='T')

    solve(a == L, temperature, bcs,
          solver_parameters=dict(linear_solver='bicgstab',
                                 preconditioner='sor'))

    return temperature


def solve_heat_elasticity_equation(geo, temperature):
    V = VectorFunctionSpace(geo.mesh, 'P', 1, dim=3)

    u = TrialFunction(V)
    v = TestFunction(V)

    bcs = [
        DirichletBC(V, Constant((0, 0, 0)), lambda x, on_boundary: geo.on_bottom_surface(x, on_boundary)),
    ]

    lambda_ = 1.0e9
    mu = 1.0e9
    rho = 922
    g = 9.8

    beta = 1.0e+4

    # Проседает под своим весом
    f = Constant((0, 0, -rho * g))

    # Нет внешнего давления
    T = Constant((0, 0, 0))

    def epsilon(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

    def sigma(u):
        return lambda_ * nabla_div(u) * Identity(3) + 2 * mu * epsilon(u)

    a = inner(sigma(u), epsilon(v)) * dx

    L = dot(f, v) * dx + dot(T, v) * ds + beta * (temperature - Constant(270)) * inner(Identity(3), epsilon(v)) * dx

    u = Function(V, name='displacement')

    solve(a == L, u, bcs)

    return u


if __name__ == '__main__':
    geo = IceIsland()

    # Сперва решаем относительно температуры
    temperature = solve_heat_equation(geo)
    File("../output/thermo_elasticity/temperature.pvd") << temperature

    # Теперь отностительно деформации
    u = solve_heat_elasticity_equation(geo, temperature)
    File("../output/thermo_elasticity/displacement.pvd") << u
