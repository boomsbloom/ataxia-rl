data {

    int<lower=1> n_obs;                             // number of observations
    int<lower=1> n_betas;                           // number of weights in softmax choice
    real outcome[n_obs];                            // outcome received on each trial
    int image[n_obs];                               // 1 = img1 was shown; 2 = img2 was shown
    int f_choice[n_obs];                            // 1 = participant pressed f; 0 = participant pressed j

}

parameters {
  
    // Subject level parameters
    vector[n_betas] beta;

}


model {
    
    real mu[n_obs];

    // inverse temperatures
    for (b in 1:n_betas) {
      beta[b] ~ normal(0,5);
    }
    
    for (n in 1:n_obs) {
        if (image[n] == 1) {
          mu[n] = beta[1];
        } else {
          mu[n] = beta[2];
        }
    }
    f_choice ~ bernoulli_logit(mu);

}

generated quantities {

    real log_lik[n_obs];
    real post_pred[n_obs];

    log_lik=rep_array(0,n_obs);
    post_pred = rep_array(0,n_obs);

    for (n in 1:n_obs) {
          if (image[n] == 1) {
            // log likelihood
            log_lik[n] = bernoulli_logit_lpmf(f_choice[n] | beta[1]);
            // posterior predictive
            post_pred[n] = bernoulli_logit_rng(beta[1]);
          } else {
            // log likelihood
            log_lik[n] = bernoulli_logit_lpmf(f_choice[n] | beta[2]);
            // posterior predictive
            post_pred[n] = bernoulli_logit_rng(beta[2]);
          }
    }
}
