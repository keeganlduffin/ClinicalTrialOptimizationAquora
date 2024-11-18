# import pyomo and the pyomo library
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

# Define continuous variables delta_mu for each covariate
model.delta_mu = Var(model.covariates, within=NonNegativeReals)

#define continouos variables delta_sigma_mod for each covariate
model.delta_sigma_mod = Var(model.covariates, within=NonNegativeReals)

#define continous variables delta_sigma for each covariate
model.delta_sigma = Var(model.covariates, within=NonNegativeReals)

# Add constraint that says that delta_mu = 1/(num_patiens)* sum of w times the difference of x_i_1 and x_i_2
def delta_mu_rule(model, i):
    return model.delta_mu[i] == sum(model.w[i, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients) / len(model.patients) #equation verified
model.delta_mu_constraint = Constraint(model.covariates, rule=delta_mu_rule)

#add constraint that says delta_sigma_mod = 1/(num_patients)* sum of w_i_s * w_i_(s+1) times the difference of x_i_1 and x_i_2
def delta_sigma_mod_rule(model, i):
    return model.delta_sigma_mod[i] == sum(model.w[i, j] * model.w[i, j+1] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients if j < len(model.patients) - 1) / len(model.patients) #equation verified
model.delta_sigma_mod_constraint = Constraint(model.covariates, rule=delta_sigma_mod_rule) #calculation is incorrect

#add constraint that says delta_sigma = 1/(num_patients)* sum of w_i_s * w_i_s times the difference of x_i_1 and x_i_2
def delta_sigma_rule(model, i):
    return model.delta_sigma[i] == sum(model.w[i, j] * model.w[i, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients) / len(model.patients) #equation verified
model.delta_sigma_constraint = Constraint(model.covariates, rule=delta_sigma_rule)

#define delta_mu_opt, delta_sigma_opt, and delta_sigma_mod_opt as parameters with 3 covariates
model.delta_mu_opt = Param(initialize=[1, 2, 3], mutable=True)
model.delta_sigma_opt = Param(initialize=[1, 2, 3], mutable=True)
model.delta_sigma_mod_opt = Param(initialize=[1, 2, 3], mutable=True)

#add a constraint that says delta_mu_opt is greater than or equal to -delta_mu and greater than or equal to delta_mu
def delta_mu_lower_bound_rule(model):
    return model.delta_mu_opt[1] >= -model.delta_mu[1]
model.delta_mu_lower_bound_constraint1 = Constraint(rule=delta_mu_lower_bound_rule)

def delta_mu_upper_bound_rule(model):
    return model.delta_mu_opt[1] >= model.delta_mu[1]
model.delta_mu_upper_bound_constraint2 = Constraint(rule=delta_mu_upper_bound_rule)

def delta_mu_lower_bound_rule(model):
    return model.delta_mu_opt[2] >= -model.delta_mu[2]
model.delta_mu_lower_bound_constraint3 = Constraint(rule=delta_mu_lower_bound_rule)

def delta_mu_upper_bound_rule(model):
    return model.delta_mu_opt[2] >= model.delta_mu[2]
model.delta_mu_upper_bound_constraint4 = Constraint(rule=delta_mu_upper_bound_rule)

def delta_mu_lower_bound_rule(model):
    return model.delta_mu_opt[3] >= -model.delta_mu[3]
model.delta_mu_lower_bound_constraint5 = Constraint(rule=delta_mu_lower_bound_rule)

def delta_mu_upper_bound_rule(model):
    return model.delta_mu_opt[3] >= model.delta_mu[3]
model.delta_mu_upper_bound_constraint6 = Constraint(rule=delta_mu_upper_bound_rule)

#add a constraint that says delta_sigma_mod_opt is greater than or equal to -delta_sigma_mod and greater than or equal to delta_sigma_mod
def delta_sigma_mod_lower_bound_rule(model):
    return model.delta_sigma_mod_opt[1] >= -model.delta_sigma_mod[1]
model.delta_sigma_mod_lower_bound_constraint1 = Constraint(rule=delta_sigma_mod_lower_bound_rule)

def delta_sigma_mod_upper_bound_rule(model):
    return model.delta_sigma_mod_opt[1] >= model.delta_sigma_mod[1]
model.delta_sigma_mod_upper_bound_constraint1 = Constraint(rule=delta_sigma_mod_upper_bound_rule)

def delta_sigma_mod_lower_bound_rule(model):
    return model.delta_sigma_mod_opt[2] >= -model.delta_sigma_mod[2]
model.delta_sigma_mod_lower_bound_constraint2 = Constraint(rule=delta_sigma_mod_lower_bound_rule)

def delta_sigma_mod_upper_bound_rule(model):
    return model.delta_sigma_mod_opt[2] >= model.delta_sigma_mod[2]
model.delta_sigma_mod_upper_bound_constraint2 = Constraint(rule=delta_sigma_mod_upper_bound_rule)

def delta_sigma_mod_lower_bound_rule(model):
    return model.delta_sigma_mod_opt[3] >= -model.delta_sigma_mod[3]
model.delta_sigma_mod_lower_bound_constraint3 = Constraint(rule=delta_sigma_mod_lower_bound_rule)

def delta_sigma_mod_upper_bound_rule(model):
    return model.delta_sigma_mod_opt[3] >= model.delta_sigma_mod[3]
model.delta_sigma_mod_upper_bound_constraint3 = Constraint(rule=delta_sigma_mod_upper_bound_rule)

#add a constraint that says delta_sigma_opt is greater than or equal to -delta_sigma and greater than or equal to delta_sigma
def delta_sigma_lower_bound_rule(model):
    return model.delta_sigma_opt[1] >= -model.delta_sigma[1]
model.delta_sigma_lower_bound_constraint1 = Constraint(rule=delta_sigma_lower_bound_rule)

def delta_sigma_upper_bound_rule(model):
    return model.delta_sigma_opt[1] >= model.delta_sigma[1]
model.delta_sigma_upper_bound_constraint1 = Constraint(rule=delta_sigma_upper_bound_rule)

def delta_sigma_lower_bound_rule(model):
    return model.delta_sigma_opt[2] >= -model.delta_sigma[2]
model.delta_sigma_lower_bound_constraint2 = Constraint(rule=delta_sigma_lower_bound_rule)

def delta_sigma_upper_bound_rule(model):
    return model.delta_sigma_opt[2] >= model.delta_sigma[2]
model.delta_sigma_upper_bound_constraint2 = Constraint(rule=delta_sigma_upper_bound_rule)

def delta_sigma_lower_bound_rule(model):
    return model.delta_sigma_opt[3] >= -model.delta_sigma[3]
model.delta_sigma_lower_bound_constraint3 = Constraint(rule=delta_sigma_lower_bound_rule)

def delta_sigma_upper_bound_rule(model):
    return model.delta_sigma_opt[3] >= model.delta_sigma[3]
model.delta_sigma_upper_bound_constraint3 = Constraint(rule=delta_sigma_upper_bound_rule)

# Define the objective function to minimize the sum of delta_mu_opt + rho * the sum of the of delta_sigma_opt + 2 * rho * the double summation from s 1 to 3 and s' = s+1 to 3 of delta_sigma_mod_opt where rho is .5
def objective_rule(model):
    return model.delta_mu_opt + .5 * model.delta_sigma_opt + model.delta_sigma_mod_opt
model.objective = Objective(rule=objective_rule)

# Create a solver (use Gurobi)
solver = SolverFactory('gurobi')

# Solve the model
solver.solve(model)

# Print the results
for i in model.covariates:
    print(f"delta_mu_{i} = {model.delta_mu[i].value}")
for i in model.patients:
    for j in model.groups:
        print(f"x_{i}_{j} = {model.x[i, j].value}")
print(f"Objective = {model.objective()}")


#model.pprint()