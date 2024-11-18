# Import pyomo and the pyomo library
from pyomo.environ import *

# Create an optimization model called clinical trial
model = ConcreteModel(name="clinical trial")

# Add a set of covariates that goes from 1 to 3
model.covariates = Set(initialize=[1, 2, 3])

# Add a set of patients that goes from 1 to n_patients (initially 6)
model.patients = Set(initialize=[1, 2, 3, 4, 5, 6])

# Add a set of groups to assign the patients to
model.groups = Set(initialize=[1, 2])

# Add a parameter w for each covariate and each patient
model.w = Param(model.covariates, model.patients, initialize={
    (1, 1): 5.8, (1, 2): 5.6, (1, 3): 7.0, (1, 4): 5.5, (1, 5): 3.8, (1, 6): 6.6,
    (2, 1): 171.8, (2, 2): 739.48, (2, 3): 51.6, (2, 4): 612.1, (2, 5): 67.1, (2, 6): 94.4,
    (3, 1): 1.22, (3, 2): 1.06, (3, 3): 1.2, (3, 4): 1.03, (3, 5): 1.09, (3, 6): 1.1
})

# Add a binary variable that assigns each patient to each group
model.x = Var(model.patients, model.groups, within=Binary)

# Add a constraint that says that each patient can only be assigned to one group
def one_group_rule(model, i):
    return sum(model.x[i, j] for j in model.groups) == 1
model.one_group = Constraint(model.patients, rule=one_group_rule)

# Add a constraint that says that the first patient needs to be assigned to the first group
def first_patient_rule(model):
    return model.x[1, 1] == 1
model.first_patient = Constraint(rule=first_patient_rule)

# Add a constraint that says that each group has to have num_patients / 2
def num_patients_rule(model, j):
    return sum(model.x[i, j] for i in model.patients) == len(model.patients) / 2
model.num_patients = Constraint(model.groups, rule=num_patients_rule)

# Define continuous variables delta_mu, delta_sigma_mod, delta_sigma for each covariate
model.delta_mu = Var(model.covariates, within=Reals)
model.delta_sigma_mod = Var(model.covariates, model.covariates, within=Reals)
model.delta_sigma = Var(model.covariates, within=Reals)

# Define auxiliary variables for absolute values using constraints
model.delta_mu_abs = Var(model.covariates, within=NonNegativeReals)
model.delta_sigma_abs = Var(model.covariates, within=NonNegativeReals)
model.delta_sigma_mod_abs = Var(model.covariates, model.covariates, within=NonNegativeReals)

# Constraints for delta_mu_abs
def delta_mu_abs_constraint_rule(model, s):
    return [
        (model.delta_mu_abs[s] >= model.delta_mu[s]),
        (model.delta_mu_abs[s] >= -model.delta_mu[s])
    ]
for s in model.covariates:
    model.add_component(f'delta_mu_abs_constraint_{s}', ConstraintList())
    for con in delta_mu_abs_constraint_rule(model, s):
        model.component(f'delta_mu_abs_constraint_{s}').add(con)

# Constraints for delta_sigma_abs
def delta_sigma_abs_constraint_rule(model, s):
    return [
        (model.delta_sigma_abs[s] >= model.delta_sigma[s]),
        (model.delta_sigma_abs[s] >= -model.delta_sigma[s])
    ]
for s in model.covariates:
    model.add_component(f'delta_sigma_abs_constraint_{s}', ConstraintList())
    for con in delta_sigma_abs_constraint_rule(model, s):
        model.component(f'delta_sigma_abs_constraint_{s}').add(con)

# Constraints for delta_sigma_mod_abs
def delta_sigma_mod_abs_constraint_rule(model, s, s_prime):
    return [
        (model.delta_sigma_mod_abs[s, s_prime] >= model.delta_sigma_mod[s, s_prime]),
        (model.delta_sigma_mod_abs[s, s_prime] >= -model.delta_sigma_mod[s, s_prime])
    ]
for s in model.covariates:
    for s_prime in model.covariates:
        if s_prime > s:
            model.add_component(f'delta_sigma_mod_abs_constraint_{s}_{s_prime}', ConstraintList())
            for con in delta_sigma_mod_abs_constraint_rule(model, s, s_prime):
                model.component(f'delta_sigma_mod_abs_constraint_{s}_{s_prime}').add(con)

# Add constraint that says that delta_mu = 1/(num_patients) * sum of w times the difference of x_i_1 and x_i_2
def delta_mu_rule(model, s):
    return model.delta_mu[s] == sum(model.w[s, i] * (model.x[i, 1] - model.x[i, 2]) for i in model.patients) / len(model.patients)
model.delta_mu_constraint = Constraint(model.covariates, rule=delta_mu_rule)

# Add constraint that says that delta_sigma = 1/(num_patients) * sum of w_s * w_s * (x_i1 - x_i2)
def delta_sigma_rule(model, s):
    return model.delta_sigma[s] == sum(model.w[s, i]**2 * (model.x[i, 1] - model.x[i, 2]) for i in model.patients) / len(model.patients)
model.delta_sigma_constraint = Constraint(model.covariates, rule=delta_sigma_rule)

# Add constraint that says that delta_sigma_mod = 1/(num_patients) * sum of w_s * w_s' * (x_i1 - x_i2)
def delta_sigma_mod_rule(model, s, s_prime):
    if s_prime > s:
        return model.delta_sigma_mod[s, s_prime] == sum(model.w[s, i] * model.w[s_prime, i] * (model.x[i, 1] - model.x[i, 2]) for i in model.patients) / len(model.patients)
    else:
        return Constraint.Skip
model.delta_sigma_mod_constraint = Constraint(model.covariates, model.covariates, rule=delta_sigma_mod_rule)

# Define the objective function to minimize the sum of delta_mu_abs + rho * the sum of delta_sigma_abs + 2 * rho * the sum of delta_sigma_mod_abs where rho is 0.5
def objective_rule(model):
    rho = 0.5
    return (
        sum(model.delta_mu_abs[s] for s in model.covariates) +
        rho * sum(model.delta_sigma_abs[s] for s in model.covariates) +
        2 * rho * sum(model.delta_sigma_mod_abs[s, s_prime] for s in model.covariates for s_prime in model.covariates if s_prime > s)
    )
model.objective = Objective(rule=objective_rule, sense=minimize)

# Create a solver (use Gurobi)
solver = SolverFactory('gurobi')

# Solve the model
solver.solve(model, tee=True)

# Print the results
for i in model.covariates:
    if model.delta_mu[i].value is not None:
        print(f"delta_mu_{i} = {model.delta_mu[i].value}")
    if model.delta_sigma[i].value is not None:
        print(f"delta_sigma_{i} = {model.delta_sigma[i].value}")
    for j in model.covariates:
        if j > i and model.delta_sigma_mod[i, j].value is not None:
            print(f"delta_sigma_mod_{i}_{j} = {model.delta_sigma_mod[i, j].value}")
for i in model.patients:
    for j in model.groups:
        if model.x[i, j].value is not None:
            print(f"x_{i}_{j} = {model.x[i, j].value}")
if model.objective() is not None:
    print(f"Objective = {model.objective()}")
