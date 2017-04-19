# coding=utf-8
from dolfin.cpp.common import has_petsc
from dolfin.cpp.function import near
from dolfin.cpp.io import File
from dolfin.cpp.la import PETScOptions
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
                                      self.R, self.R), 256)

    def on_bottom_surface(self, x, on_boundary):
        return on_boundary and near(x[2], 0)

    def on_upper_surface(self, x, on_boundary):
        return on_boundary and near(x[2], self.H)

    def on_underwater_side_surface(self, x, on_boundary):
        return on_boundary and 0 < x[2] < self.h

    def on_above_water_side_surface(self, x, on_boundary):
        return on_boundary and self.h < x[2] < self.H


def solve_heat_equation():
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

    File("../output/thermo_elasticity/temperature.pvd") << temperature

    return temperature


if __name__ == '__main__':
    geo = IceIsland()

    # Сперва решаем относительно температуры
    temperature = solve_heat_equation()
