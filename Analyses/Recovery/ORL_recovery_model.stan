// ORL Model
data {
  int<lower = 1> N; // Number of total observations
  real o[N]; // outcome for each subject on each trial
  real sign[N]; // sign for frequency updating for each subject on each trial
  int x[N]; // choice for each subject on each trial
}

// Set starting value at zero
transformed data {
  vector[8] initV;
  initV = rep_vector(0.0,8);
}

parameters {
 // individual parameters 
 real<lower=0, upper = 1> Arew;
 real<lower=0, upper = 1> Apun;
 real<lower=0, upper = 5> K;
 real wf;
 real wp;
 
}
 
model {
  vector[8] EF; // Expected frequency
  vector[8] EV; // Expedcted value
  vector[8] PS; // Perseverance
  vector[8] util; // Combined 'choice value'
  vector[8] PE_EFall; // Prediction error frequency, unchosen
  vector[8] PE_EVall; // Prediction error value, unchosen 
    
  real PE_EF; // Prediction error for expected frequency
  real PE_EV; // Prediciton errror for expected value
  real EF_chosen; // Expected frequency for chosen deck
  real EV_chosen; // Expected value for chosen deck
  real K_tr; // K transformed 
    
  // individual priors
  Arew ~ uniform(0, 1);
  Apun ~ uniform(0,1);
  K ~ uniform(0,5);
  wp ~ normal(0,1);
  wf ~ normal(0,1);
      
  // Set starting values to zero
  EF = initV;
  EV = initV;
  PS = initV;
  util = initV;
  K_tr = pow(3, K)-1; // Transform K 
    
    
  for (t in 1:N){
    // Make choice 
    x[t] ~ categorical_logit(util); 
      
    // Compute prediction errors
    PE_EV = o[t] - EV[x[t]];
    PE_EF = sign[t] - EF[x[t]];
    PE_EFall = -sign[t]/7 - EF; // Compute counterfactual prediction error
    
    // Store EF and EV for chosen deck
    EF_chosen = EF[x[t]];
    EV_chosen = EV[x[t]];
      
    if (o[t] >= 0){
      // Update EF for all decks
      EF += Apun * PE_EFall;
      // Update chosen deck
      EF[x[t]] = EF_chosen + Arew*PE_EF;
      EV[x[t]] = EV_chosen + Arew*PE_EV;
    } else{
        
      EF += Arew*PE_EFall;
        
      EF[x[t]] = EF_chosen + Apun*PE_EF;
      EV[x[t]] = EV_chosen + Apun*PE_EV;
    }
      
    // Update perseverance
    PS[x[t]] = 1;
    PS /= (1+ K_tr);
      
    // Calculate utility
    util = EV + EF*wf + PS * wp;
  }
}




