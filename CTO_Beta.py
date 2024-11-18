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
model.w = Param(model.covariates,
                model.patients,
                initialize={
                    (1, 1): 58, (1, 2): 56, (1, 3): 70, (1, 4): 55, (1, 5): 38, (1, 6): 66,
                    (2, 1): 1718, (2, 2): 7394.8, (2, 3): 516, (2, 4): 6121, (2, 5): 671, (2, 6): 944,
                    (3, 1): 12.2, (3, 2): 10.6, (3, 3): 12, (3, 4): 10.3, (3, 5): 10.9, (3, 6): 11})

# Add a parameter rho that is equal to 0.5
model.rho = Param(initialize=0.5)

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

# Define continuous variables delta_mu, delta_sigma_mod, and delta_sigma for each covariate
model.delta_mu = Var(model.covariates, within=Reals)

# Define variable sigma for every pair of 2 covariates
model.delta_sigma = Var(model.covariates, model.covariates, within=Reals)

# Only consider upper triangular combinatios of covariates for delta_sigma
def delta_sigma_rule(model, s, s_prime):
    # If s_prime is smaller than s, skip the constraint
    if s_prime < s:
        return Constraint.Skip
    # Otherwise, enforce that delta_sigma[s, s_prime] is equal 1/n_patients(sum of w_i_s * w_i_s_prime times the difference of x_i_1 and x_i_2)
    return model.delta_sigma[s, s_prime] == sum(model.w[s, i] * model.w[s_prime, i] * (model.x[i, 1] - model.x[i, 2]) for i in model.patients) / len(model.patients)

model.delta_sigma_constraint = Constraint(model.covariates, model.covariates, rule=delta_sigma_rule)

# Add constraint that calculates delta_mu
def delta_mu_rule(model, s):
    return model.delta_mu[s] == sum(model.w[s, i] * (model.x[i, 1] - model.x[i, 2]) for i in model.patients) / len(model.patients)
model.delta_mu_constraint = Constraint(model.covariates, rule=delta_mu_rule)

# Add variable for absolute value of delta_mu
model.delta_mu_abs = Var(model.covariates, within=NonNegativeReals)

# Add constraint that calculates the absolute value of delta_mu
def delta_mu_abs_rule1(model, s):
    return model.delta_mu[s] <= model.delta_mu_abs[s]
model.delta_mu_abs_constraint1 = Constraint(model.covariates, rule=delta_mu_abs_rule1)

def delta_mu_abs_rule2(model, s):
    return -model.delta_mu[s] <= model.delta_mu_abs[s]
model.delta_mu_abs_constraint2 = Constraint(model.covariates, rule=delta_mu_abs_rule2)

# Add variable for absolute value of delta_sigma
model.delta_sigma_abs = Var(model.covariates, model.covariates, within=NonNegativeReals)

# Add constraint that calculates the absolute value of delta_sigma
def delta_sigma_abs_rule1(model, s, s_prime):
    return model.delta_sigma[s, s_prime] <= model.delta_sigma_abs[s, s_prime]
model.delta_sigma_abs_constraint1 = Constraint(model.covariates, model.covariates, rule=delta_sigma_abs_rule1)

def delta_sigma_abs_rule2(model, s, s_prime):
    return -model.delta_sigma[s, s_prime] <= model.delta_sigma_abs[s, s_prime]
model.delta_sigma_abs_constraint2 = Constraint(model.covariates, model.covariates, rule=delta_sigma_abs_rule2)



# # Define indexed parameters delta_mu_opt, delta_sigma_opt, delta_sigma_mod_opt for each covariate (mutable)
# model.delta_mu_opt = Param(model.covariates, initialize={1: 0, 2: 0, 3: 0}, mutable=True, within=Reals)
# model.delta_sigma_opt = Param(model.covariates, initialize={1: 0, 2: 0, 3: 0}, mutable=True, within=Reals)
# model.delta_sigma_mod_opt = Param(model.covariates, initialize={1: 0, 2: 0, 3: 0}, mutable=True, within=Reals)

# #add a constraint that says delta_mu_opt is greater than or equal to -delta_mu and greater than or equal to delta_mu
# def delta_mu_lower_bound_rule(model):
#     return model.delta_mu_opt[1] >= -model.delta_mu[1]
# model.delta_mu_lower_bound_constraint1 = Constraint(rule=delta_mu_lower_bound_rule)

# def delta_mu_upper_bound_rule(model):
#     return model.delta_mu_opt[1] >= model.delta_mu[1]
# model.delta_mu_upper_bound_constraint2 = Constraint(rule=delta_mu_upper_bound_rule)

# def delta_mu_lower_bound_rule(model):
#     return model.delta_mu_opt[2] >= -model.delta_mu[2]
# model.delta_mu_lower_bound_constraint3 = Constraint(rule=delta_mu_lower_bound_rule)

# def delta_mu_upper_bound_rule(model):
#     return model.delta_mu_opt[2] >= model.delta_mu[2]
# model.delta_mu_upper_bound_constraint4 = Constraint(rule=delta_mu_upper_bound_rule)

# def delta_mu_lower_bound_rule(model):
#     return model.delta_mu_opt[3] >= -model.delta_mu[3]
# model.delta_mu_lower_bound_constraint5 = Constraint(rule=delta_mu_lower_bound_rule)

# def delta_mu_upper_bound_rule(model):
#     return model.delta_mu_opt[3] >= model.delta_mu[3]
# model.delta_mu_upper_bound_constraint6 = Constraint(rule=delta_mu_upper_bound_rule)

# #add a constraint that says delta_sigma_mod_opt is greater than or equal to -delta_sigma_mod and greater than or equal to delta_sigma_mod
# def delta_sigma_mod_lower_bound_rule(model):
#     return model.delta_sigma_mod_opt[1] >= -model.delta_sigma_mod[1]
# model.delta_sigma_mod_lower_bound_constraint1 = Constraint(rule=delta_sigma_mod_lower_bound_rule)

# def delta_sigma_mod_upper_bound_rule(model):
#     return model.delta_sigma_mod_opt[1] >= model.delta_sigma_mod[1]
# model.delta_sigma_mod_upper_bound_constraint1 = Constraint(rule=delta_sigma_mod_upper_bound_rule)

# def delta_sigma_mod_lower_bound_rule(model):
#     return model.delta_sigma_mod_opt[2] >= -model.delta_sigma_mod[2]
# model.delta_sigma_mod_lower_bound_constraint2 = Constraint(rule=delta_sigma_mod_lower_bound_rule)

# def delta_sigma_mod_upper_bound_rule(model):
#     return model.delta_sigma_mod_opt[2] >= model.delta_sigma_mod[2]
# model.delta_sigma_mod_upper_bound_constraint2 = Constraint(rule=delta_sigma_mod_upper_bound_rule)

# def delta_sigma_mod_lower_bound_rule(model):
#     return model.delta_sigma_mod_opt[3] >= -model.delta_sigma_mod[3]
# model.delta_sigma_mod_lower_bound_constraint3 = Constraint(rule=delta_sigma_mod_lower_bound_rule)

# def delta_sigma_mod_upper_bound_rule(model):
#     return model.delta_sigma_mod_opt[3] >= model.delta_sigma_mod[3]
# model.delta_sigma_mod_upper_bound_constraint3 = Constraint(rule=delta_sigma_mod_upper_bound_rule)

# #add a constraint that says delta_sigma_opt is greater than or equal to -delta_sigma and greater than or equal to delta_sigma
# def delta_sigma_lower_bound_rule(model):
#     return model.delta_sigma_opt[1] >= -model.delta_sigma[1]
# model.delta_sigma_lower_bound_constraint1 = Constraint(rule=delta_sigma_lower_bound_rule)

# def delta_sigma_upper_bound_rule(model):
#     return model.delta_sigma_opt[1] >= model.delta_sigma[1]
# model.delta_sigma_upper_bound_constraint1 = Constraint(rule=delta_sigma_upper_bound_rule)

# def delta_sigma_lower_bound_rule(model):
#     return model.delta_sigma_opt[2] >= -model.delta_sigma[2]
# model.delta_sigma_lower_bound_constraint2 = Constraint(rule=delta_sigma_lower_bound_rule)

# def delta_sigma_upper_bound_rule(model):
#     return model.delta_sigma_opt[2] >= model.delta_sigma[2]
# model.delta_sigma_upper_bound_constraint2 = Constraint(rule=delta_sigma_upper_bound_rule)

# def delta_sigma_lower_bound_rule(model):
#     return model.delta_sigma_opt[3] >= -model.delta_sigma[3]
# model.delta_sigma_lower_bound_constraint3 = Constraint(rule=delta_sigma_lower_bound_rule)

# def delta_sigma_upper_bound_rule(model):
#     return model.delta_sigma_opt[3] >= model.delta_sigma[3]
# model.delta_sigma_upper_bound_constraint3 = Constraint(rule=delta_sigma_upper_bound_rule)

# Define the objective function to minimize the sum of delta_mu_opt, delta_sigma_opt, and delta_sigma_mod_opt
# def objective_rule(model):
#     return sum(model.delta_mu_opt[i] + model.rho * model.delta_sigma_opt[i] + 2 * model.rho * model.delta_sigma_mod_opt[i] for i in model.covariates)
def objective_rule(model):
    return sum(model.delta_mu_abs[s] for s in model.covariates) + model.rho * sum(model.delta_sigma_abs[s,s] for s in model.covariates) + 2 * model.rho * sum(sum(model.delta_sigma_abs[s, s_prime] for s_prime in model.covariates if s_prime > s) for s in model.covariates)
model.objective = Objective(rule=objective_rule, sense=minimize)


# Create a solver (use Gurobi)
solver = SolverFactory('gurobi')

# Solve the model
solver.solve(model)

model.pprint()

# Print the results
for i in model.covariates:
    print(f"delta_mu_abs_{i} = {model.delta_mu_abs[i].value}")
    for j in model.covariates:
        print(f"delta_sigma_abs_{i}_{j} = {model.delta_sigma_abs[i, j].value}")
for i in model.patients:
    for j in model.groups:
        print(f"x_{i}_{j} = {model.x[i, j].value}")
print(f"Objective = {model.objective()}")

