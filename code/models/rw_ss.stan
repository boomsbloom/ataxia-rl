data {

    int<lower=1> n_obs;                             // number of observations
    int<lower=1> n_betas;                           // number of weights in softmax choice
    real outcome[n_obs];                            // outcome received on each trial
    int image[n_obs];                               // 1 = img1 was shown; 2 = img2 was shown
    int f_choice[n_obs];                            // 1 = participant pressed f; 0 = participant pressed j

}

parameters {
  
    // Subject level parameters
    real<lower=0, upper=1> alpha;
    vector[n_betas] beta;

}


transformed parameters {

    // Hybrid Pearce-Hall RW model
    real<lower=0, upper=1> Q1[n_obs,2];       // Q values
    real<lower=0, upper=1> Q2[n_obs,2];       // Q values
    real qdiff[n_obs];
    real delta[n_obs];                        // prediction error

    Q1 = rep_array(0.0,n_obs,2);
    Q2 = rep_array(0.0,n_obs,2);
    // set initial values of Q and alpha
    Q1[1,1] = 0.5; // j choice
    Q1[1,2] = 0.5; // f choice
    Q2[1,1] = 0.5; // j choice
    Q2[1,2] = 0.5; // f choice
    for (n in 1:n_obs) {

        // Compute prediction error
        if (image[n] == 1) {
            delta[n] = outcome[n] - Q1[n,f_choice[n]+1];
            qdiff[n] = Q1[n,2] - Q1[n,1];
        } else {
            delta[n] = outcome[n] - Q2[n,f_choice[n]+1];
            qdiff[n] = Q2[n,2] - Q2[n,1];
        }

        if (n < n_obs) {

            if (image[n] == 1) {
                Q1[n+1,f_choice[n]+1] = Q1[n,f_choice[n]+1] + alpha * delta[n];
                Q1[n+1,abs(f_choice[n]-2)] = Q1[n,abs(f_choice[n]-2)];
                Q2[n+1,1] = Q2[n,1];
                Q2[n+1,2] = Q2[n,2];
            } else {
                Q2[n+1,f_choice[n]+1] = Q2[n,f_choice[n]+1] + alpha * delta[n];
                Q2[n+1,abs(f_choice[n]-2)] = Q2[n,abs(f_choice[n]-2)];
                Q1[n+1,1] = Q1[n,1];
                Q1[n+1,2] = Q1[n,2];
            }

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
        if (image[n] == 1) {
          mu[n] = (beta[1] + beta[3] * qdiff[n]);
        } else {
          mu[n] = (beta[2] + beta[4] * qdiff[n]);
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
            log_lik[n] = bernoulli_logit_lpmf(f_choice[n] | beta[1] + beta[3] * qdiff[n]);
            // posterior predictive
            post_pred[n] = bernoulli_logit_rng(beta[1] + beta[3] * qdiff[n]);
          } else {
            // log likelihood
            log_lik[n] = bernoulli_logit_lpmf(f_choice[n] | beta[2] + beta[4] * qdiff[n]);
            // posterior predictive
            post_pred[n] = bernoulli_logit_rng(beta[2] + beta[4] * qdiff[n]);
          }
    }
}
