// ORL Model
data {
  int<lower = 1> N; // Number of total observations
  int<lower=1> s;  // Number of subjects
  int<lower=1> Tsubj[s]; // Number of trials for each subject
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
  //Subject level raw parameters
 vector<lower=0, upper = 1>[s] Arew;
 vector<lower=0, upper = 1>[s] Apun;
 vector<lower=0, upper = 5>[s] K;
 vector[s] wf;
 vector[s] wp;
 
}
 
model {
  
  int pos; // Keeps track of position to seperate data by subject
  pos = 1; // Starting position
  
  // individual priors
  Arew ~ uniform(0, 1);
  Apun ~ uniform(0,1);
  K ~ uniform(0,5);
  wp ~ normal(0,1);
  wf ~ normal(0,1);
  
  for (i in 1:s){
    // empty vectors
    int x_i[Tsubj[i]];
    real o_i[Tsubj[i]];
    real sign_i[Tsubj[i]];
      
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
    
    // slice data and assing to empty vectors
    x_i = segment(x, pos, Tsubj[i]);
    o_i = segment(o, pos, Tsubj[i]);
    sign_i = segment(sign, pos, Tsubj[i]);
    
    // Set starting values to zero
    EF = initV;
    EV = initV;
    PS = initV;
    util = initV;
    
    
    for (t in 1:Tsubj[i]){
      // Make choice 
      x_i[t] ~ categorical_logit(util); 
      
      // Compute prediction errors
      PE_EV = o_i[t] - EV[x_i[t]];
      PE_EF = sign_i[t] - EF[x_i[t]];
      PE_EFall = -sign_i[t]/7 - EF; // Compute counterfactual prediction error
      
      // Store EF and EV for chosen deck
      EF_chosen = EF[x_i[t]];
      EV_chosen = EV[x_i[t]];
      
      if (o_i[t] >= 0){
        // Update EF for all decks
        EF += Apun[i] * PE_EFall;
        // Update chosen deck
        EF[x_i[t]] = EF_chosen + Arew[i]*PE_EF;
        EV[x_i[t]] = EV_chosen + Arew[i]*PE_EV;
      } else{
        
        EF += Arew[i]*PE_EFall;
        
        EF[x_i[t]] = EF_chosen + Apun[i]*PE_EF;
        EV[x_i[t]] = EV_chosen + Apun[i]*PE_EV;
      }
      
      // Update perseverance
      PS[x_i[t]] = 1;
      PS /= (1+ K[i]);
      
      // Calculate utility
      util = EV + EF*wf[i] + PS * wp[i];
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
      real sign_i[Tsubj[i]];
      
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
      log_lik[i] = 0; // initial log-likelihood for the subject
    
      // slice data and assing to empty vectors
      x_i = segment(x, pos, Tsubj[i]);
      o_i = segment(o, pos, Tsubj[i]);
      sign_i = segment(sign, pos, Tsubj[i]);
    
      // Set starting values to zero
      EF = initV;
      EV = initV;
      PS = initV;
      util = initV;
    
    
      for (t in 1:Tsubj[i]){
        // Log likelihood calculation
        log_lik[i] += categorical_logit_lpmf(x_i[t]|util);
        
        // Make choice 
        y_pred[t+pos-1] = categorical_rng(softmax(util)); 
      
        // Compute prediction errors
        PE_EV = o_i[t] - EV[x_i[t]];
        PE_EF = sign_i[t] - EF[x_i[t]];
        PE_EFall = -sign_i[t]/7 - EF; // Compute counterfactual prediction error
      
        // Store EF and EV for chosen deck
        EF_chosen = EF[x_i[t]];
        EV_chosen = EV[x_i[t]];
      
        if (o_i[t] >= 0){
          // Update EF for all decks
          EF += Apun[i] * PE_EFall;
          // Update chosen deck
          EF[x_i[t]] = EF_chosen + Arew[i]*PE_EF;
          EV[x_i[t]] = EV_chosen + Arew[i]*PE_EV;
        } else{
        
          EF += Arew[i]*PE_EFall;
        
          EF[x_i[t]] = EF_chosen + Apun[i]*PE_EF;
          EV[x_i[t]] = EV_chosen + Apun[i]*PE_EV;
        }
      
        // Update perseverance
        PS[x_i[t]] = 1;
        PS /= (1+ K[i]);
      
        // Calculate utility
        util = EV + EF*wf[i] + PS * wp[i];
      }
    pos = pos + Tsubj[i]; // Specifies starting position for next subject
   }
  }
}
