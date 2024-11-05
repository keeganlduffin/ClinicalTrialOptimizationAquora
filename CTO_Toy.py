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

# Add a constraint that says that the firest patient needs to be assigned to the first group
def first_patient_rule(model):
    return model.x[1, 2] == 0
model.first_patient = Constraint(rule=first_patient_rule)

# Add a constraint that says that each group has to have num_patients / 2
def num_patients_rule(model, j):
    return sum(model.x[i, j] for i in model.patients) == len(model.patients) / 2
model.num_patients = Constraint(model.groups, rule=num_patients_rule)

# Define continuous variables delta_mu for each covariate
model.delta_mu = Var(model.covariates, within=NonNegativeReals)

#define continuous variables delta_sigma_mod for each covariate
model.delta_sigma_mod = Var(model.covariates, within=NonNegativeReals)

#define continuous variables delta_sigma for each covariate
model.delta_sigma = Var(model.covariates, within=NonNegativeReals)

#add constraint that says delta_sigma_2 = 1/(num_patients)* sum of w_i_s * w_i_(s+1) times the difference of x_i_1 and x_i_2; sigma_ss'
# def delta_sigma_mod_rule(model, i):
#     return model.delta_sigma_mod[i] == sum(model.w[i, j] * model.w[i, j+1] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients if j < len(model.patients) - 1) / len(model.patients)
# model.delta_sigma_mod_constraint = Constraint(model.covariates, rule=delta_sigma_mod_rule)

def delta_sigma_mod_rule_ub(model, i):
    delta_sigma_mod_ub = sum(model.w[i, j] * model.w[k, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients for k in range(i+1,3)) / len(model.patients)
    return delta_sigma_mod_ub <= model.delta_sigma_mod[i]
def delta_sigma_mod_rule_lb(model, i):
    delta_sigma_mod_lb = -sum(model.w[i, j] * model.w[k, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients for k in range(i+1,3)) / len(model.patients)
    return delta_sigma_mod_lb <= model.delta_sigma_mod[i]

model.delta_sigma_mod_constraint_ub = Constraint(model.covariates, rule=delta_sigma_mod_rule_ub)
model.delta_sigma_mod_constraint_lb = Constraint(model.covariates, rule=delta_sigma_mod_rule_lb)


#add a constraint that says delta_mu must be greater than the equation used to calculate it or the negative of the equation used to calculate it
def delta_mu_constraint_rule_ub(model, i):
    delta_mu_ub = sum(model.w[i, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients) / len(model.patients)
    return delta_mu_ub <= model.delta_mu[i] 
def delta_mu_constraint_rule_lb(model, i):
    delta_mu_lb = -sum(model.w[i, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients) / len(model.patients)
    return delta_mu_lb <= model.delta_mu[i] 

model.delta_mu_constraint_ub = Constraint(model.covariates, rule=delta_mu_constraint_rule_ub)
model.delta_mu_constraint_lb = Constraint(model.covariates, rule=delta_mu_constraint_rule_lb)

#add constraint that says delta_sigma = 1/(num_patients)* sum of w_i_s * w_i_s times the difference of x_i_1 and x_i_2; sigma_ss
def delta_sigma_rule_ub(model, i):
    delta_sigma_ub = sum(model.w[i, j] * model.w[i, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients) / len(model.patients)
    return delta_sigma_ub <= model.delta_sigma[i]
def delta_sigma_rule_lb(model, i):
    delta_sigma_lb = -sum(model.w[i, j] * model.w[i, j] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients) / len(model.patients)
    return delta_sigma_lb <= model.delta_sigma[i]
model.delta_sigma_constraint_ub = Constraint(model.covariates, rule=delta_sigma_rule_ub)
model.delta_sigma_constraint_lb = Constraint(model.covariates, rule=delta_sigma_rule_lb)

#add a constraint that says delta_sigma_mod must be greater than the equation used to calculate it or the negative of the equation used to calculate it
# def delta_sigma_mod_constraint_rule(model, i):
#     return model.delta_sigma_mod[i] >= sum(model.w[i, j] * model.w[i, j+1] * (model.x[j, 1] - model.x[j, 2]) for j in model.patients if j < len(model.patients) - 1) / len(model.patients)
# model.delta_sigma_mod_constraint = Constraint(model.covariates, rule=delta_sigma_mod_constraint_rule)

# Define the objective to minimize the sum of delta_mu + rho * the sum of the of delta_sigma_s_s + 2 * rho * the double summation from s 1 to 3 and s' = s+1 to 3 of delta_sigma_2_s_s' where rho is .5
model.rho = 0.5
def objective_rule(model):
    return sum(model.delta_mu[i] for i in model.covariates) + model.rho * sum(model.delta_sigma[i] for i in model.covariates) + 2 * model.rho * sum(model.delta_sigma_mod[i] for i in model.covariates)
model.objective = Objective(rule=objective_rule, sense=minimize)

# Create a solver (use Gurobi)
solver = SolverFactory('gurobi', solver_io='python')

# Solve the model
solver.solve(model)

# Print the results
for i in model.covariates:
    print(f"delta_mu_{i} = {model.delta_mu[i].value}")
for i in model.patients:
    for j in model.groups:
        print(f"x_{i}_{j} = {model.x[i, j].value}")
print(f"Objective = {model.objective()}")


model.pprint()