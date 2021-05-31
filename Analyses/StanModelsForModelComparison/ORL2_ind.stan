// ORL Model
data {
  int<lower = 1> N; // Number of total observations
  int<lower= 1> s; // number of subjects
  int<lower = 1> Tsubj[s]; // Number of trials for each subject
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
 vector<lower = 0, upper = 1>[s] Afreq; // Frequency learning
 vector<lower = 0, upper = 1>[s] Aval; // Value learning
 vector<lower = 0, upper= 1>[s] Ahab; // Habitual learning
 vector[s] wfreq; // Frequency weighting
 vector[s] whab; // Habitual weighting
 
}
 
model {
  
  int pos; // Keeps track of position to seperate data by subject
  pos = 1; // Starting position
  
  // individual priors
  Afreq ~ uniform(0, 1);
  Aval ~ uniform(0,1);
  Ahab ~ uniform(0,1);
  wfreq ~ normal(0,1);
  whab ~ normal(0,1); 
  
  for (i in 1:s){
    // empty vectors
    int x_i[Tsubj[i]];
    real o_i[Tsubj[i]];
    real freq_i[Tsubj[i]];
  
    vector[8] EF; // Expected frequency
    vector[8] EV; // Expected value
    vector[8] HAB; // Habitual signal
    vector[8] util; // Combined 'choice value'
      
    real PE_EF; // Prediction error for expected frequency
    real PE_EV; // Prediciton errror for expected value
    real EF_chosen; // Expected frequency for chosen deck
    real EV_chosen; // Expected value for chosen deck
    real HAB_chosen; // Container for the habitual value for each deck 

    // slice data and assing to empty vectors
    x_i = segment(x, pos, Tsubj[i]);
    o_i = segment(o, pos, Tsubj[i]);
    freq_i = segment(freq, pos, Tsubj[i]);    
 
    // Set starting values to zero
    EF = initV;
    EV = initV;
    HAB = initV;
    util = initV;
    
    for (t in 1:Tsubj[i]){
      // Make choice 
      x_i[t] ~ categorical_logit(util); 
        
      // Compute prediction errors
      PE_EV = o_i[t] - EV[x_i[t]];
      PE_EF = freq_i[t] - EF[x_i[t]];
      
      // Store EF, EV and HAB for chosen deck
      EF_chosen = EF[x_i[t]];
      EV_chosen = EV[x_i[t]];
      HAB_chosen = HAB[x_i[t]];
        
      // Update EF and EV only for chosen decks
      EF[x_i[t]] = EF_chosen + Afreq[i]*PE_EF;
      EV[x_i[t]] = EV_chosen + Aval[i]*PE_EV;
  
      // Update habitual signal for all decks
      HAB = HAB + Ahab[i] * (initV - HAB); // Update all slots as they were unchosen
      HAB[x_i[t]] = HAB_chosen + Ahab[i]*(1- HAB_chosen); // Update the chosen decks and override value 
        
      // Calculate utility
      util = EV + EF*wfreq[i] + HAB* whab[i];
    }
    pos = pos + Tsubj[i]; // Specifies starting position for next subject
  }
}

generated quantities{
  // Log likelihood for each subject
  real log_lik[s];
  
  // Predicted choice
  real y_pred[N];
  
  int pos; // Keeps track of position to seperate data by subject
  pos = 1; // Starting position
  
  {
   for (i in 1:s){
    // empty vectors
    int x_i[Tsubj[i]];
    real o_i[Tsubj[i]];
    real freq_i[Tsubj[i]];
  
    vector[8] EF; // Expected frequency
    vector[8] EV; // Expected value
    vector[8] HAB; // Habitual signal
    vector[8] util; // Combined 'choice value'
      
    real PE_EF; // Prediction error for expected frequency
    real PE_EV; // Prediciton errror for expected value
    real EF_chosen; // Expected frequency for chosen deck
    real EV_chosen; // Expected value for chosen deck
    real HAB_chosen; // Container for the habitual value for each deck 

    // slice data and assing to empty vectors
    x_i = segment(x, pos, Tsubj[i]);
    o_i = segment(o, pos, Tsubj[i]);
    freq_i = segment(freq, pos, Tsubj[i]);    
 
    // Set starting values to zero
    EF = initV;
    EV = initV;
    HAB = initV;
    util = initV;
    log_lik[i] = 0; // initial log-likelihood for the subject

    
    for (t in 1:Tsubj[i]){
      // Log likelihood calculation
      log_lik[i] += categorical_logit_lpmf(x_i[t]|util);
        
      // Make choice 
      y_pred[t+pos-1] = categorical_rng(softmax(util)); 
      
      // Compute prediction errors
      PE_EV = o_i[t] - EV[x_i[t]];
      PE_EF = freq_i[t] - EF[x_i[t]];
      
      // Store EF, EV and HAB for chosen deck
      EF_chosen = EF[x_i[t]];
      EV_chosen = EV[x_i[t]];
      HAB_chosen = HAB[x_i[t]];
        
      // Update EF and EV only for chosen decks
      EF[x_i[t]] = EF_chosen + Afreq[i]*PE_EF;
      EV[x_i[t]] = EV_chosen + Aval[i]*PE_EV;
  
      // Update habitual signal for all decks
      HAB = HAB + Ahab[i] * (initV - HAB); // Update all slots as they were unchosen
      HAB[x_i[t]] = HAB_chosen + Ahab[i]*(1- HAB_chosen); // Update the chosen decks and override value 
        
      // Calculate utility
      util = EV + EF*wfreq[i] + HAB* whab[i];
      }
    pos = pos + Tsubj[i]; // Specifies starting position for next subject
   }
  }
}


