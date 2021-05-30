data {
  int<lower = 1> N; // Number of total observations
  real o[N]; // outcome for each subject on each trial. The real model treats losses and wins seperately!
  int x[N]; // choice for each subject on each trial
  int<lower = 1> n_games; //Number of games in the dataset
}

// Set starting value at zero
transformed data {
  vector[n_games] initV;
  initV = rep_vector(0.00,n_games);
}

parameters {
 // Model parameters
 real<lower= 0, upper =1> theta;
 real<lower = 0, upper =1> delta;
 real<lower = 0, upper= 1> alpha;
 real phi;
 real<lower=0, upper = 5> C;
}


model {
  
  vector[n_games] Exploit;
  vector[n_games] Explore;
    
  real value;
  real Exploit_chosen;
  
  // individual priors
  theta ~ uniform(0, 1);
  delta ~ uniform(0,1);
  C ~ uniform(0,5);
  phi ~ normal(0,1);
  alpha ~ uniform(0,1);

  // Set starting value for explore 
  Exploit = initV;
  Explore = initV;
    
  // Transform beta
    
  for (t in 1:N){

    x[t] ~ categorical_logit((Explore+Exploit)*C);
      
    if (o[t]< 0){
      value = - pow(fabs(o[t]), theta);
    }else{
      value = pow(o[t], theta);
    }
      
    // Select chosen machines
    Exploit_chosen = Exploit[x[t]];
    
    // Update exploit signal for chosen and unchosen machines
    Exploit = Exploit*delta; // For all machines
    Exploit[x[t]] = Exploit_chosen*delta + value;
    
    // Update sequential explore signal
    Explore += alpha*(phi-Explore);
    Explore[x[t]] = 0;
  }
}

