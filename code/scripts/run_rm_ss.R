args <- commandArgs(trailingOnly = TRUE)

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Set variables
basedir = "/burg/dslab/users/jdn2133/ataxia/"
model_name = args[1]
group_name = args[2]
subj_id = args[3]
n_betas = 3
n_iters = 2000

# Load and prepare the data
if (group_name == "pat") {
  stan_data = read.csv(sprintf("%sdata/patientMapData4Stan.csv",basedir))
} else if (group_name == "ctrl") {
  stan_data = read.csv(sprintf("%sdata/controlMapData4Stan.csv",basedir))
} else if (group_name == "ideal") {
  stan_data = read.csv(sprintf("%sdata/idealMapData4Stan.csv",basedir))
}
stan_data = stan_data[stan_data$sub_factor==subj_id,]

n_obs = nrow(stan_data)
model_data = list("n_obs" = n_obs, # number of observations
                  "n_betas" = n_betas, # number of predictors
                  "outcome" = stan_data$outcome, # Outcome variable
                  "image" = stan_data$image, # Which image was shown
                  "f_choice" = stan_data$f_chosen, # Whether f was pressed or not
                  "stay"= stan_data$stay) # Whether the participant stayed on the same choice as the last trial

# Fit and save the model
sm = sprintf("%smodels/%s.stan",basedir,model_name)
out_file = sprintf("%sfit_models/%s_fit_%s_%s.rds",basedir,model_name,group_name,subj_id)
fit <- stan(file = sm, data = model_data, iter=n_iters, seed=999)
saveRDS(fit, file = out_file)

# Extract main parameters from the fit model for easier access outside of R
paramdir = sprintf("%sfit_models/",basedir)
if (model_name == "ph_ss") {
  params_extract_only = c("Q1","Q2","kappa","eta","alpha")
  model_params = c()
} else if (model_name == "rw") {
  params_extract_only = c("Q1","Q2","alpha")
  model_params = c("a1","a2")
} else if (model_name == "wsls") {
  params_extract_only = c()
  model_params = c()
} else if (model_name == "rand") {
  params_extract_only = c()
  model_params = c()
}

param2extract = append(c("beta","log_lik","post_pred"),model_params)
#param2extract = append(c("beta_mu","u","log_lik","post_pred"),model_params)
param2extract = append(param2extract, params_extract_only)
extracted_params = extract(fit,pars=param2extract)
saveRDS(extracted_params, file=sprintf("%s%s_%s_%s_extracted_params.rds",paramdir,model_name,group_name,subj_id))

# Run and save LOO
loo = loo(fit, moment_match=TRUE)
saveRDS(loo, file=sprintf("%s%s_%s_%s_loo.rds",paramdir,model_name,group_name,subj_id))