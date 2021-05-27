# This algorithm estimates a cognitive profile for each player based on slot machine gambling data following a specific data format (see Master's Thesis).
# The algorithm is written in the programming language R but uses Stan to estimate cognitive parameters.
# Before running the function make sure that you have a C++ Toolchain installed. Please see https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started for instructions.

# As a user you need to input the data to the algorithm. Further you may specify estimation method, model, minimum number of sessions for inclusion of the player. 
# Lastly you may decide whether you want to save the full posterior, the 95 % High Density Interval and whether progress should be printed.
# Notice that keeping the full posterior results in very large datafiles.
# Notice that Variational Bayes (VB) may provide inaccurate estimates but is significantly faster than the No-U-Turn Sampler (NUTS), which is the default of this algorithm.

CreateCognitiveProfile = function(data, method = "NUTS", model = "ORL_HAB", min_ses = 80, keep_full_posterior = F, output_HDI = T, print_progress = T){ 
  
  # Check if the chosen model is available
  available_models = c("PVLD", "VSE", "ORL", "ORL_HAB", "ORL_HAB_FREQ")
  
  # Model parameter list
  if (model == "ORL_HAB") param_list = c("Arew", "Apun", "Ahab", "wf", "whab")
  if (model == "ORL_HAB_FREQ") param_list = c("Afreq", "Aval", "Ahab", "wfreq", "whab")
  if (model == "ORL") param_list = c("Arew", "Apun", "K", "wf", "wp")
  if (model == "PVLD") param_list = c("A", "w", "a", "c")
  if (model == "VSE") param_list = c("theta", "delta", "alpha", "phi", "c")
  
  if (model %in% available_models == FALSE){
    print(paste("Model not available. Choose a model from the following list:", paste(unlist(available_models), collapse = ', '), sep = " "))
  }
  
  # Check that the required packages are installed. If not the packages are installed automatically.
  if(!require(pacman)) install.packages("pacman")
  pacman::p_load(rstan, bayestestR, parallel, stringr, rlist)
  
  res_dat = data.frame(ID = character(), param1 = numeric(), param2 = numeric(), param3 = numeric(), param4 =  numeric(), param5 = numeric()) # Empty dataframe
  if (model == "PVLD") res_dat = res_dat[,-6]
  
  if (output_HDI == TRUE) {
    HDI_data = data.frame(ID = character(), param1 = numeric(), param2 = numeric(), param3 = numeric(), param4 =  numeric(), param5 = numeric())
    if (model == "PVLD") HDI_dat = HDI_dat[,-6]
  }
  
  if (keep_full_posterior == TRUE){
    full_posterior = data.frame(ID = character(), param1 = numeric(), param2 = numeric(), param3 = numeric(), param4 =  numeric(), param5 = numeric())
    if (model == "PVLD") full_posterior = full_posterior[,-6]
  }
  
  n = 1 # Iterations counter
  
  # Only select players with more than the minimum number of sessions
  counts = data %>% count(ID)
  IDS = counts$ID[counts$n >= min_ses,]
  
  # Chose how many cores to use for NUTS or load model for Variational Bayes
  if (method == "NUTS"){      
    n_cores_available = detectCores() 
    n_cores= ifelse(n_cores_available > 4, 4, n_cores_available-1) 
  }else if (method == "VB"){
    model = stan_model(str_glue("Models/{model}.stan"))
  }
  
  for (id in IDS){ # Loop through all players
    
    # Prepare data  
    temp = data[data$ID == id,]
    
    N = length(temp$ID)
    o = temp$outcome/temp$sesion_length 
    x = temp$game
    
    if (model == "ORL_HAB_FREQ"){
      freq = temp$win_freq
      dats = list(N = N, o = o, freq = freq, x = x)
    } else {
      sign = sign(o)
      dats = list(N = N, o = o, sign = sign, x = x)
    }
    
    # Fit model & extract samples  
    if (method == "NUTS"){
      fit=stan(file="Models/{model}.stan", data = dats, cores = n_cores, chains = 4, warmup = 1500, iter = 4000)
    } else if (method == "VB") {
      fit = vb(model, dats, tol_rel_obj = 0.001, output_samples = 10000)
    } else {
      print("The chosen method is not supported by the algorithm. Please input 'NUTS' for MCMC sampling and 'VB' for Variational Inference.")
      break
    }
      
    ext_samp = rstan::extract(fit)
      
    # Estimate MAP & append to dataframe
    param1_MAP = map_estimate(ext_samp[1])
    param2_MAP = map_estimate(ext_samp[2])
    param3_MAP = map_estimate(ext_samp[3])
    param4_MAP = map_estimate(ext_samp[4])
    if (model != "PVLD") param5_MAP = map_estimate(ext_samp[5])
      
    if (model == "PVLD"){
      results = c(id, param1_MAP, param2_MAP, param3_MAP, param4_MAP)
    } else {
      results = c(id, param1_MAP, param2_MAP, param3_MAP, param4_MAP, param5_MAP)
    }
    
    
    res_dat[n,] = results
    
    # Estimate 95 % HDI. May be omitted by the user.
    if (output_HDI == TRUE){
      param1_hdi = hdi(ext_samp[1], ci = 0.95)
      param2_hdi = hdi(ext_samp[2], ci = 0.95)
      param3_hdi = hdi(ext_samp[3], ci = 0.95)
      param4_hdi = hdi(ext_samp[4], ci = 0.95)
      if (model != "PVLD") param5_hdi = hdi(ext_samp[5], ci = 0.95)
      
      if (model == "PVLD"){
        HDI_temp = c(id, param1_hdi, param2_hdi, param3_hdi, param4_hdi)
      } else {
        HDI_temp = c(id, param1_hdi, param2_hdi, param3_hdi, param4_hdi, param5_hdi)
      }
    
      HDI_data[n,] = HDI_temp  
    }
    
    # Save full posterior if this is specified by the user
    if (keep_full_posterior == TRUE){
      if (model != "PVLD") full_posterior[1+(n-1)*10000:n*10000,] = data.frame(ID = rep(id, 10000), param1 = ext_samp[1], param2 = ext_samp[2], param3 = ext_samp[3], param4 = ext_samp[4], param5 = ext_samp[5])
      if (model == "PVLD") full_posterior[1+(n-1)*10000:n*10000,] = data.frame(ID = rep(id, 10000), param1 = ext_samp[1], param2 = ext_samp[2], param3 = ext_samp[3], param4 = ext_samp[4])
    }
    
    # Print progress at intervals of 10 if the user has specified this.  
    if(print_progress == TRUE & n %% 10==0){ 
      print(paste(n, "out of", IDS, "players analysed.", sep = " "))
    }
    n=n+1
  }
  
  if (keep_full_posterior == F & output_HDI == F){
    names(res_dat)[-1] <- param_list # Add parameter names to the dataframe
    return(res_dat) # Return a dataframe if the user has only requires the MAP
  } else {
    output_list = list(res_dat)
    if (output_HDI == T){
      names(HDI_dat)[-1] <- param_list 
      output_list = list.append(output_list, HDI_data)
    }
    if (keep_full_posterior == T){
      names(full_posterior)[-1] <- param_list
      output_list = list.append(output_list, full_posterior)
    }
    return(output_list) # Return a list of dataframes if full_posterior or HDI is requested
  }
  
  print(paste("All", n, "players have been analysed", sep = " "))
  
}
