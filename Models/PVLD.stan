data {
  int<lower=1> N;
  int x[N];
  real o[N];
}
transformed data {
  vector[8] initV;
  initV  = rep_vector(0.0, 8);
}
parameters {
  // Subject-level raw parameters 
  real<lower= 0, upper= 1> A;
  real<lower= 0, upper =5> w;
  real<lower = 0, upper = 1> a;
  real<lower = 0, upper= 5> c;
}

model {
  // Define values
  vector[8] ev;
  real curUtil;     // utility of curFb
  real theta;       // theta = 3^c - 1

  // individual parameters
  A ~ uniform(0, 1);
  w ~ uniform(0, 5);
  a ~ uniform(0, 1);
  c ~ uniform(0, 5);

  // Initialize values
//  theta = pow(3, c) -1;
  ev = initV; // initial ev values

  for (t in 1:N) {
    // softmax choice
    x[t] ~ categorical_logit(c * ev);

    if (o[t] >= 0) {  // x(t) >= 0
      curUtil = pow(o[t], A);
    } else {                  // x(t) < 0
      curUtil = -1 * w * pow(-1 * o[t], A);
    }

    // delta
    ev[x[t]] += a * (curUtil - ev[x[t]]);
  }
}
