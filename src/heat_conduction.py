# coding=utf-8
from dolfin import *
from dolfin.cpp.io import File
from dolfin.cpp.mesh import Point
from mshr import *


class HeatSolver:
    def __init__(self):
        self.mesh = generate_mesh(Cylinder(Point(0.0, 0.0, 10.0),
                                           Point(0.0, 0.0, 0.0),
                                           100, 100), 64)
        self.time_step = 0.1
        self.n_steps = 100

        self.function_space = FunctionSpace(self.mesh, 'P', 2)

        self.dirichlet_bc = self._create_dirichlet_bc()
        self.rhs_function = Constant(0.0)

        self.vtk_file = File("./output/heat_problem/solution.pvd")

    def run(self):
        u_prev = project(self.dirichlet_function, self.function_space)

        u = TrialFunction(self.function_space)
        v = TestFunction(self.function_space)

        F = (u * v * dX
             + self.time_step * dot(grad(u), grad(v)) * dX
             - (u_prev + self.time_step * self.rhs_function) * v * dX)
        a, L = lhs(F), rhs(F)

        u = Function(self.function_space)
        t = 0
        for n in range(self.n_steps):
            t += self.time_step
            self.dirichlet_bc.t = t

            solve(a == L, u, self.dirichlet_bc)
            u_prev.assign(u)

            self.vtk_file << u

    def _create_dirichlet_bc(self):
        def dirichlet_boundary(x, on_boundary):
            return on_boundary

        self.dirichlet_function = Expression('c * exp(-l * t)',
                                             degree=2, c=12.0, l=12.0, t=0)
        return DirichletBC(self.function_space,
                           self.dirichlet_function,
                           dirichlet_boundary)


if __name__ == '__main__':
    solver = HeatSolver()
    solver.run()
