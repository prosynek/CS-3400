import unittest

import numpy as np

from numerical_differentiation import NumericalDifferentiation

class ParabolicCostFunction:
    def cost(self, params):
        """
        Implements the parabola:
        
        y = 5(x-3)**2 + 4
        
        The minima should be at x = 3
        """
        return 5 * (params - 3) * (params - 3) + 4
        
    def gradient(self, params):
        """
        THIS METHOD IS NOT NEEDED FOR THE OPTIMIZER.
        
        Just here to help with unit tests.
        """
        return 10 * (params - 3)
        
class ParabaloidCostFunction:
    def cost(self, params):
        """
        Implements the parabola:
        
        z = (x-2)**2 + (y-3)**2
        
        The minima should be at x = 2, y = 3
        """
        return (params[0] - 2)**2 + (params[1] - 3)**2
        
    def gradient(self, params):
        """
        THIS METHOD IS NOT NEEDED FOR THE OPTIMIZER.
        
        Just here to help with unit tests.
        """
        dx = 2 * (params[0] - 2)
        dy = 2 * (params[1] - 3)
        return np.array([dx, dy])
        
class TestOptimizer(unittest.TestCase):
    NUMERIC_DIFF_DELTA = 1e-5
    GRADIENT_TOL = 1e-2
    
    def test_gradient_parabola(self):
        """
        Tests the gradient method using a parabolic cost function with a single parameter.
        """
        nd = NumericalDifferentiation(self.NUMERIC_DIFF_DELTA)
        cost = ParabolicCostFunction()
        
        # cost function is 1D therefore params should be a 1D array
        params = np.zeros(1)
        
        # try at x = -1
        params[0] = -1
        numeric_gradient = nd.gradient(cost, params)
        analytic_gradient = cost.gradient(params)

        # expect 1D array
        self.assertEqual(len(numeric_gradient.shape), 1)
        self.assertEqual(numeric_gradient.shape[0], 1)
        
        diff = numeric_gradient - analytic_gradient
        norm = np.linalg.norm(diff)
        
        self.assertLess(norm, self.GRADIENT_TOL)
        
    def test_gradient_paraboloid(self):
        """
        Tests the gradient method using a paraboloid cost function with two parameters.
        """
        nd = NumericalDifferentiation(self.NUMERIC_DIFF_DELTA)
        cost = ParabaloidCostFunction()        
        
        # cost function is 2D therefore params should be a 2D array
        params = np.zeros(2)
        
        # try at x = -1, y = 1
        params[0] = -1
        params[1] = 1
        numeric_gradient = nd.gradient(cost, params)
        analytic_gradient = cost.gradient(params)

        # expect a 1D array
        self.assertEqual(len(numeric_gradient.shape), 1)
        self.assertEqual(numeric_gradient.shape[0], 2)
        
        diff = numeric_gradient - analytic_gradient
        norm = np.linalg.norm(diff)
        
        self.assertLess(norm, self.GRADIENT_TOL)

if __name__ == "__main__":
    unittest.main()