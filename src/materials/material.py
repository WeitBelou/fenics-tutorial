# coding=utf-8
from fenics import *


def epsilon(u):
    """
    Тензор деформации
    :param u: смещение
    :return: 
    """
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)


class IsotropicMaterial:
    def __init__(self, lambda_, mu, beta):
        self.lambda_ = lambda_
        self.mu = mu
        self.beta = beta

    def thermal_expansion(self):
        """
        Температурное расширение
        :return: 
        """
        return - Identity(3) * self.beta

    def stiffness(self, u):
        """
        Жёсткость
        :return: 
        """
        eps = epsilon(u)
        return self.lambda_ * sum(diag(eps)) * Identity(3) + 2 * self.mu * eps

    def stress(self, u, theta):
        """
        Возвращает тензор напряжений
        :param u: смещение
        :param theta: температура
        :return: 
        """
        return self.stiffness(u) - self.thermal_expansion() * theta
