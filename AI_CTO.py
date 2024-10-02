import numpy as np
from pyomo.environ import *

# Problem parameters
n = 100  # number of patients
r = 3    # number of covariates
rho = 0.5  # weight for the second moment

# Assume w is the matrix of normalized patient covariates (n x r)
# Here we use random data for demonstration, replace with actual data
np.random.seed(42)
w = np.random.randn(n, r)

# Pyomo model
model = ConcreteModel()

# Decision variables: x[i, p] = 1 if patient i is in group p, else 0
model.x = Var(range(n), range(2), domain=Binary)

# Constraints
# 1. Each patient must be assigned to exactly one group
def one_group_rule(model, i):
    return model.x[i, 0] + model.x[i, 1] == 1
model.one_group_constraint = Constraint(range(n), rule=one_group_rule)

# 2. Each group must have exactly n/2 patients
def group_size_rule(model, p):
    return sum(model.x[i, p] for i in range(n)) == n // 2
model.group_size_constraint = Constraint(range(2), rule=group_size_rule)

# Symmetry breaking (optional): Fix one assignment to prevent trivial solutions
model.x[0, 1].fix(0)

# Objective function: minimize the discrepancy in first and second moments
def discrepancy_rule(model):
    # First moments (means)
    mu_diff = sum(abs(sum(w[i, s] * (model.x[i, 0] - model.x[i, 1]) for i in range(n)))
                  for s in range(r))
    
    # Second moments (variances and covariances)
    var_diff = sum(abs(sum(w[i, s]**2 * (model.x[i, 0] - model.x[i, 1]) for i in range(n)))
                   for s in range(r))
    
    cov_diff = sum(abs(sum(w[i, s] * w[i, s_prime] * (model.x[i, 0] - model.x[i, 1]) for i in range(n)))
                   for s in range(r) for s_prime in range(s+1, r))

    # Objective: sum of differences in means, variances, and covariances
    return mu_diff + rho * (var_diff + 2 * cov_diff)

model.obj = Objective(rule=discrepancy_rule, sense=minimize)

# Solve the problem
solver = SolverFactory('glpk')  # You can replace with other solvers like 'cbc' or 'gurobi'
solver.solve(model, tee=True)

# Retrieve results
group1 = [i for i in range(n) if model.x[i, 0].value == 1]
group2 = [i for i in range(n) if model.x[i, 1].value == 1]

print("Group 1:", group1)
print("Group 2:", group2)
