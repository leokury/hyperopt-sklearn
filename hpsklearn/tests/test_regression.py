try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
from hyperopt import rand, tpe
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components


class TestRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.X_train = np.random.randn(100, 2)
        self.Y_train = self.X_train[:, 0] * 2
        self.X_test = np.random.randn(100, 2)
        self.Y_test = self.X_test[:, 0] * 2

def create_function(reg_fn):
    def test_regressor(self):
        model = hyperopt_estimator(
            regressor=reg_fn('regressor', n_jobs=-1),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=10,
            verbose=True
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_regressor.__name__ = 'test_{0}'.format(reg_fn.__name__)
    return test_regressor


# List of regressors to test
regressors = [
    components.lgbm_regression,
]


# Create unique methods with test_ prefix so that nose can see them
for reg in regressors:
    setattr(
        TestRegression,
        'test_{0}'.format(reg.__name__),
        create_function(reg)
    )



if __name__ == '__main__':
    unittest.main()

# -- flake8 eof
