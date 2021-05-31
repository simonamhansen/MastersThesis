data {
  int<lower = 1> N; // Number of total observations
  int<lower=1> s;  // Number of subjects
  int<lower=1> Tsubj[s]; // Number of trials for each subject
  real o[N]; // outcome for each subject on each trial. The real model treats losses and wins seperately!
  int x[N]; // choice for each subject on each trial
}

transformed data {
  vector[8] initV;
  initV  = rep_vector(0.0, 8);
}
parameters {
  // Subject-level parameters
  vector<lower= 0, upper= 1>[s] A;
  vector<lower= 0, upper =5>[s] w;
  vector<lower = 0, upper = 1>[s] a;
  vector<lower = 0, upper= 5>[s] c;
}

model {
  int pos; // Keeps track of position for ragged data structure
  pos = 1; // Starting position for the first subject

  // individual priors. Totally uninformative
  A ~ uniform(0, 1);
  w ~ uniform(0, 5);
  a ~ uniform(0, 1);
  c ~ uniform(0, 5);

  for (i in 1:s){
    // Empty containers
    int x_i[Tsubj[i]];
    real o_i[Tsubj[i]];
    
    vector[8] ev;
    real curUtil;     // utility 
    real theta;
    
    x_i =  segment(x, pos, Tsubj[i]);
    o_i = segment(o, pos, Tsubj[i]);
    
    // Initialize values
    //  theta = pow(3, c) -1;
    ev = initV; // initial ev values
  
    for (t in 1:Tsubj[i]) {
      // softmax choice
      x_i[t] ~ categorical_logit(c[i] * ev);
  
      if (o_i[t] >= 0) {  // x(t) >= 0
        curUtil = pow(o_i[t], A[i]);
      } else {                  // x(t) < 0
        curUtil = -1 * w[i] * pow(-1 * o_i[t], A[i]);
      }
  
      // delta
      ev[x_i[t]] += a[i] * (curUtil - ev[x_i[t]]);
    }
    pos = pos + Tsubj[i]; // Specifies starting position for next subject
  }
}

generated quantities {
  // Log likelihood of each subjects choice given parameter estimates and choices up until now.
  real log_lik[s];
  
  // Predicted choice
  real y_pred[N];  
  
  int pos; // Keeps track of position to seperate data by subject
  pos = 1; // Starting position
  
  {
    for (i in 1:s){
    // Empty containers
    int x_i[Tsubj[i]];
    real o_i[Tsubj[i]];
    
    vector[8] ev;
    real curUtil;     // utility 
    real theta;
    log_lik[i] = 0; 
    
    x_i =  segment(x, pos, Tsubj[i]);
    o_i = segment(o, pos, Tsubj[i]);
    
    // Initialize values
    //  theta = pow(3, c) -1;
    ev = initV; // initial ev values
  
    for (t in 1:Tsubj[i]) {
      // Log-likelihood calculation
      log_lik[i] += categorical_logit_lpmf(x_i[t] | c[i]*ev);
      
      // softmax choice
      y_pred[t+pos-1] = categorical_rng(softmax(c[i] * ev));
  
      if (o_i[t] >= 0) {  // x(t) >= 0
        curUtil = pow(o_i[t], A[i]);
      } else {                  // x(t) < 0
        curUtil = -1 * w[i] * pow(-1 * o_i[t], A[i]);
      }
  
      // delta
      ev[x_i[t]] += a[i] * (curUtil - ev[x_i[t]]);
    }
    pos = pos + Tsubj[i]; // Specifies starting position for next subject
  }
    
  }
  
}
