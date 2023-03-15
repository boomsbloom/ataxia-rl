args <- commandArgs(trailingOnly = TRUE)

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Set variables
basedir = "/burg/dslab/users/jdn2133/ataxia/"
model_name = args[1]
group_name = args[2]
subj_id = args[3]
n_iters = 2000

if (model_name == "hybrid_ph_ss") {
  outcome = "outcome"
  n_betas = 4
} else if (model_name == "only_ph_ss") {
  outcome = "outcome"
  n_betas = 2
} else if (model_name == "only_ep_ss") {
  outcome = "outcome"
  n_betas = 2
}

# Load and prepare the data
if (group_name == "pat") {
  stan_data = read.csv(sprintf("%sdata/patientHybridData4Stan.csv",basedir))
  n_subjects = nrow(unique(stan_data[c("sub_factor")]))
} else if (group_name == "ctrl") {
  stan_data = read.csv(sprintf("%sdata/controlHybridData.csv",basedir))
  n_subjects = nrow(unique(stan_data[c("sub_factor")]))
} else if (group_name == "ideal") {
  stan_data = read.csv(sprintf("%sdata/idealHybridData.csv",basedir))
  n_subjects = nrow(unique(stan_data[c("sub_factor")]))
}
stan_data = stan_data[stan_data$sub_factor==subj_id,]

n_obs = nrow(stan_data)
stan_data$red_chosen[stan_data$red_chosen == 1] = 0
stan_data$red_chosen[stan_data$red_chosen == 2] = 1
noise = sd(stan_data$outcome_coded)
outcomeRange = c(-0.5,0.5)
model_data = list("n_obs" = n_obs, # number of observations
                  "n_betas" = n_betas, # number of predictors
                  "noise" = noise, # Standard deviation of the outcomes
                  "outcomeRange" = outcomeRange, # Range of the outcomes
                  "outcome" = stan_data[,outcome], # Outcome variable
                  "old_red_value" = stan_data$old_value_4_HBI, # Value of old object on red deck
                  "old_deck" = stan_data$old_deck, # If the deck is old, which deck is old
                  "red_choice" = stan_data$red_chosen) # Whether the red deck was chosen or not

# Fit and save the model
sm = sprintf("%smodels/%s.stan",basedir,model_name)
out_file = sprintf("%sfit_models/%s_fit_%s_%s.rds",basedir,model_name,group_name,subj_id)
fit <- stan(file = sm, data = model_data, iter=n_iters, seed=999)
saveRDS(fit, file = out_file)

# Extract main parameters from the fit model for easier access outside of R
paramdir = sprintf("%sfit_models/",basedir)
if (model_name == "ph_ss") {
  param2extract = c("Q1","Q2","kappa","eta","alpha","beta","log_lik","post_pred")
}

paramdir = sprintf("%sfit_models/",basedir)
extracted_params = extract(fit,pars=param2extract)
saveRDS(extracted_params, file=sprintf("%s%s_%s_%s_extracted_params.rds",paramdir,model_name,group_name,subj_id))

# Run and save LOO
loo = loo(fit, moment_match=TRUE)
saveRDS(loo, file=sprintf("%s%s_%s_%s_loo.rds",paramdir,model_name,group_name,subj_id))