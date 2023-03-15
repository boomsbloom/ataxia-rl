data {

    int<lower=1> n_obs;                             // number of observations
    int<lower=1> n_betas;                           // number of weights in softmax choice
    real outcome[n_obs];                            // outcome received on each trial
    real old_red_value[n_obs];                      // value of old object
    real old_deck[n_obs];                           // which deck the old object appeared on, 0.5 = old on red; -0.5 = old on blue
    int red_choice[n_obs];                          // 1 = participant chose red, 0 = participant chose blue

}

parameters {
  
    // Subject level parameters
    real<lower=0, upper=1> alpha;
    vector[n_betas] beta;

}


transformed parameters {

    real<lower=0, upper=1> Q[n_obs,2];       // Q values
    real delta[n_obs];                       // prediction error
    real stay[n_obs];                        // wsls

    Q = rep_array(0.0,n_obs,2);
    Q[1,1] = 0.5; // blue choice
    Q[1,2] = 0.5; // red choice
    for (n in 1:n_obs) {

        // RW model

        // Compute prediction error
        delta[n] = outcome[n] - Q[n,red_choice[n]+1];

        if (n < n_obs) {
          
            // Update value of chosen option for next trial
            Q[n+1,red_choice[n]+1] = Q[n,red_choice[n]+1] + alpha * delta[n];

            // Leave value of unchosen option alone
            Q[n+1,abs(red_choice[n]-2)] = Q[n,abs(red_choice[n]-2)];

        }

    }

}

model {
  
    real mu[n_obs];

    // inverse temperatures
    for (b in 1:n_betas) {
      beta[b] ~ normal(0,5);
    }

    // learning rate
    alpha ~ beta(1, 1);
    
    for (n in 1:n_obs) {
        mu[n] = (beta[1] * (Q[n,2] - Q[n,1]) +
                 beta[2] * old_red_value[n] +
                 beta[3] * old_deck[n]);
    }
    red_choice ~ bernoulli_logit(mu);
    
}

generated quantities {

    real log_lik[n_obs];
    real post_pred[n_obs];

    log_lik=rep_array(0,n_obs);
    post_pred = rep_array(0,n_obs);

    for (n in 1:n_obs) {
      // log likelihood
      log_lik[n] = bernoulli_logit_lpmf(red_choice[n] | beta[1] * (Q[n,2] - Q[n,1]) +
                                                        beta[2] * old_red_value[n] +
                                                        beta[3] * old_deck[n]);
      // posterior predictive
      post_pred[n] = bernoulli_logit_rng(beta[1] * (Q[n,2] - Q[n,1]) +
                                         beta[2] * old_red_value[n] +
                                         beta[3] * old_deck[n]);
    }
}
