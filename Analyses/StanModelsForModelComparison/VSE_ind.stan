data {
  int<lower = 1> N; // Number of total observations
  int<lower=1> s;  // Number of subjects
  int<lower=1> Tsubj[s]; // Number of trials for each subject
  real o[N]; // outcome for each subject on each trial. The real model treats losses and wins seperately!
  int x[N]; // choice for each subject on each trial
}

// Set starting value at zero
transformed data {
  vector[8] initV;
  initV = rep_vector(0.00,8);
}

parameters {
 // Subject level raw parameters
 vector<lower= 0, upper =1>[s] theta;
 vector<lower=0, upper=1>[s] delta;
 vector<lower=0, upper=1>[s] alpha;
 vector[s] phi;
 vector<lower= 0, upper = 5>[s] beta;
}


model {
  
  int pos; // Keeps track of position to seperate data by subject
  pos = 1; // Starting position
  

  // individual priors
  theta ~ uniform(0, 1);
  delta ~ uniform(0,1);
  beta ~ uniform(0,5);
  phi ~ normal(0,1);
  alpha ~ uniform(0,1);
  
  for (i in 1:s){
    // empty vectors
    int x_i[Tsubj[i]];
    real o_i[Tsubj[i]];
    
    vector[8] Exploit;
    vector[8] Explore;
    
    real value;
    real Exploit_chosen;
    real C;
    
    // slice data and assing to empty vectors
    x_i = segment(x, pos, Tsubj[i]);
    o_i = segment(o, pos, Tsubj[i]);
    
    // Set starting value for explore 
    Exploit = initV;
    Explore = initV;
    
    // Transform beta
    C = pow(3, beta[i])-1;
    
    for (t in 1:Tsubj[i]){

      x_i[t] ~ categorical_logit((Explore+Exploit)*C);
      
      if (o_i[t]< 0){
        value = - pow(fabs(o_i[t]), theta[i]);
      }else{
        value = pow(o_i[t], theta[i]);
      }
      
      // Select chosen machines
      Exploit_chosen = Exploit[x_i[t]];
    
      // Update exploit signal for chosen and unchosen machines
      Exploit = Exploit*delta[i]; // For all machines
      Exploit[x_i[t]] = Exploit_chosen*delta[i] + value;
    
      // Update sequential explore signal
      Explore += alpha[i]*(phi[i]-Explore);
      Explore[x_i[t]] = 0;
      
    }
    pos = pos + Tsubj[i]; // Specifies starting position for next subject
  }
}

generated quantities{
  // Log likelihood of each subjects choice given parameter estimates and choices up until now.
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
    
    vector[8] Exploit;
    vector[8] Explore;
    
    real C; 
    real value;
    real Exploit_chosen;
    
    // slice data and assing to empty vectors
    x_i = segment(x, pos, Tsubj[i]);
    o_i = segment(o, pos, Tsubj[i]);
    
    // Set starting value for explore 
    Exploit = initV;
    Explore = initV;
    log_lik[i] = 0; // initial log-likelihood for the subject
    
    // Transform beta
    C = pow(3, beta[i])-1;
    
    for (t in 1:Tsubj[i]){
      log_lik[i] += categorical_logit_lpmf(x_i[t]| (Explore+Exploit)*C); 
      
      y_pred[t+pos-1] = categorical_rng(softmax((Explore+Exploit)*beta[i]));
      
      if (o_i[t]< 0){
        value = - pow(fabs(o_i[t]), theta[i]);
      }else{
        value = pow(o_i[t], theta[i]);
      }
      
      // Select chosen machines
      Exploit_chosen = Exploit[x_i[t]];
    
      // Update exploit signal for chosen and unchosen machines
      Exploit = Exploit*delta[i]; // For all machines
      Exploit[x_i[t]] = Exploit_chosen*delta[i] + value;
    
      // Update sequential explore signal
      Explore += alpha[i]*(phi[i]-Explore);
      Explore[x_i[t]] = 0;
      
    }
    
    pos = pos + Tsubj[i]; // Specifies starting position for next subject
  }
  }
}
