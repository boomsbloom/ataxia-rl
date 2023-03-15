data {

    int<lower=1> n_obs;                             // number of observations
    int<lower=1> n_betas;                           // number of weights in softmax choice
    real outcome[n_obs];                            // outcome received on each trial
    int red_choice[n_obs];                            // 1 = participant chose red; 0 = participant chose blue

}

parameters {
  
    // Subject level parameters
    real beta;

}


model {
    
    real mu[n_obs];

    // inverse temperature
    beta ~ normal(0,5);
    
    red_choice ~ bernoulli_logit(beta);

}

generated quantities {

    real log_lik[n_obs];
    real post_pred[n_obs];

    log_lik=rep_array(0,n_obs);
    post_pred = rep_array(0,n_obs);

    for (n in 1:n_obs) {
      // log likelihood
      log_lik[n] = bernoulli_logit_lpmf(red_choice[n] | beta);
      // posterior predictive
      post_pred[n] = bernoulli_logit_rng(beta);
    }
}
