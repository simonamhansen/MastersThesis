// ORL Model
data {
  int<lower = 1> N; // Number of total observations
  real o[N]; // outcome for each subject on each trial
  real freq[N]; // sign for frequency updating for each subject on each trial
  int x[N]; // choice for each subject on each trial
}

// Set starting value at zero
transformed data {
  vector[8] initV;
  initV = rep_vector(0.0,8);
}

parameters {
 real<lower = 0, upper = 1> Afreq; // Frequency learning
 real<lower = 0, upper = 1> Aval; // Value learning
 real<lower = 0, upper= 1> Ahab; // Habitual learning
 real wfreq; // Frequency weighting
 real whab; // Habitual weighting
 
}
 
model {
  vector[8] EF; // Expected frequency
  vector[8] EV; // Expected value
  vector[8] HAB; // Habitual signal
  vector[8] util; // Combined 'choice value'
    
  real PE_EF; // Prediction error for expected frequency
  real PE_EV; // Prediciton errror for expected value
  real EF_chosen; // Expected frequency for chosen deck
  real EV_chosen; // Expected value for chosen deck
  real HAB_chosen; // Container for the habitual value for each deck 
    
  // individual priors
  Afreq ~ uniform(0, 1);
  Aval ~ uniform(0,1);
  Ahab ~ uniform(0,1);
  wfreq ~ normal(0,1);
  whab ~ normal(0,1);
      
  // Set starting values to zero
  EF = initV;
  EV = initV;
  HAB = initV;
  util = initV;
    
    
  for (t in 1:N){
    // Make choice 
    x[t] ~ categorical_logit(util); 
      
    // Compute prediction errors
    PE_EV = o[t] - EV[x[t]];
    PE_EF = freq[t] - EF[x[t]];
    
    // Store EF, EV and HAB for chosen deck
    EF_chosen = EF[x[t]];
    EV_chosen = EV[x[t]];
    HAB_chosen = HAB[x[t]];
      
    // Update EF and EV only for chosen decks
    EF[x[t]] = EF_chosen + Afreq*PE_EF;
    EV[x[t]] = EV_chosen + Aval*PE_EV;

    // Update habitual signal for all decks
    HAB = HAB + Ahab * (initV - HAB); // Update all slots as they were unchosen
    HAB[x[t]] = HAB_chosen + Ahab*(1- HAB_chosen); // Update the chosen decks and override value 
      
    // Calculate utility
    util = EV + EF*wfreq + HAB* whab;
  }
}




