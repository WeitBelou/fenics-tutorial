# coding=utf-8
from dolfin.cpp.io import File
from dolfin.cpp.mesh import Point
from fenics import *
from mshr import generate_mesh, Cylinder


class ElasticitySover:
    def __init__(self):
        self._create_mesh()
        self._create_function_space()
        self._create_boundary_conditions()
        self._create_rhs_function()

    def _create_boundary_conditions(self):
        self.dirichlet_bc = self._create_dirichlet_bc()

    def _create_function_space(self):
        self.function_space = VectorFunctionSpace(self.mesh, 'P', 1)

    def _create_rhs_function(self):
        rho = 1
        gamma = 10
        self.rhs_function = Constant((0, 0, -rho * gamma))

    def _create_mesh(self):
        L = 0.2
        R = 1.0
        self.mesh = generate_mesh(Cylinder(Point(0.0, 0.0, L),
                                           Point(0.0, 0.0, 0.0),
                                           R, R), 64)

    def _create_dirichlet_bc(self):
        def clamped_boundary(x, on_boundary):
            tol = 1e-4
            return on_boundary and x[2] < tol

        return DirichletBC(self.function_space,
                           Constant((0.0, 0.0, 0.0)),
                           clamped_boundary)

    @staticmethod
    def _epsilon(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

    @staticmethod
    def _sigma(u):
        lambda_ = 1.25
        mu = 1.0
        return lambda_ * nabla_div(u) * Identity(3) + 2 * mu * ElasticitySover._epsilon(u)

    def run(self):
        u = TrialFunction(self.function_space)
        v = TestFunction(self.function_space)

        T = Constant((0.0, 0.0, 0.0))

        a = inner(ElasticitySover._sigma(u),
                  ElasticitySover._epsilon(v)) * dX
        L = dot(self.rhs_function, v) * dX + dot(T, v) * ds

        u = Function(self.function_space)
        solve(a == L, u, self.dirichlet_bc)

        vtk_file = File('../output/elasticity/solution.pvd')

        vtk_file << u


if __name__ == '__main__':
    solver = ElasticitySover()
    solver.run()
