// ORL Model
data {
  int<lower = 1> N; // Number of total observations
  real o[N]; // outcome for each subject on each trial
  real sign[N]; // sign for frequency updating for each subject on each trial
  int x[N]; // choice for each subject on each trial
  int<lower = 1> n_games; //Number of games in the dataset
}

// Set starting value at zero
transformed data {
  vector[n_games] initV;
  initV = rep_vector(0.0,n_games);
}

parameters {
 // individual parameters 
 real<lower=0, upper = 1> Arew;
 real<lower=0, upper = 1> Apun;
 real<lower=0, upper = 1> Ahab;
 real wf;
 real whab;
 
}
 
model {
  vector[n_games] EF; // Expected frequency
  vector[n_games] EV; // Expedcted value
  vector[n_games] HAB; // Perseverance
  vector[n_games] util; // Combined 'choice value'
  vector[n_games] PE_EFall; // Prediction error frequency, unchosen
  vector[n_games] PE_EVall; // Prediction error value, unchosen 
    
  real PE_EF; // Prediction error for expected frequency
  real PE_EV; // Prediciton errror for expected value
  real EF_chosen; // Expected frequency for chosen deck
  real EV_chosen; // Expected value for chosen deck
  real HAB_chosen; // Container for the habitual value for each deck 

    
  // individual priors
  Arew ~ uniform(0, 1);
  Apun ~ uniform(0,1);
  Ahab ~ uniform(0,1);
  whab ~ normal(0,1);
  wf ~ normal(0,1);
      
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
    PE_EF = sign[t] - EF[x[t]];
    PE_EFall = -sign[t]/(n_games-1) - EF; // Compute counterfactual prediction error
    
    // Store EF and EV for chosen deck
    EF_chosen = EF[x[t]];
    EV_chosen = EV[x[t]];
    HAB_chosen = HAB[x[t]];

      
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
      
    // Update habitual signal for all decks
    HAB = HAB + Ahab * (initV - HAB); // Update all slots as they were unchosen
    HAB[x[t]] = HAB_chosen + Ahab*(1- HAB_chosen); // Update the chosen decks and override value 
      
    // Calculate utility
    util = EV + EF*wf + HAB * whab;
  }
}




