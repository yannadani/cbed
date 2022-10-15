suppressMessages(library(pcalg))
suppressMessages(library(gRbase))

adj2dag = function(adj_mat){
  p = dim(adj_mat)[1]
  e = empty.graph(as.character(1:p))
  arc_set = which(adj_mat != 0, arr.ind=TRUE)
  arc_set = apply(arc_set, 2, as.character)
  colnames(arc_set) = c("from", "to")
  arcs(e) = arc_set
  return(e)
}

avg_adj_mat = function(adj_mats){
  p = dim(adj_mats[[1]])[1]
  avg_mat = matrix(0, p, p)
  N = length(adj_mats)
  for(i in 1:N){
    inc_mat = adj_mats[[i]]
    inc_mat[inc_mat != 0] = 1
    avg_mat = avg_mat + inc_mat
  }
  return(avg_mat / N)
}

bootstrap_gies = function(n_boot, data, interventions, seed, path='../data/TEMP_DAGS/'){
  set.seed(seed)
  p = ncol(data)
  colnames(data) = as.character(1:p)
  adj_mats = list()

  for(i in 1:n_boot){
    boot_indcs = bootstrap_indcs(interventions)
    W = run_gies(data[boot_indcs, ], interventions[boot_indcs])
    colnames(W) = as.character(1:p)
    adj_mats[[i]] = W
    write.csv(W, paste(path, i, sep=''), row.names=FALSE)
  }
  return(adj_mats)
}

bootstrap_indcs = function(interventions){
  n = length(interventions)
  indcs = c()
  possible_interventions = unique(interventions)
  for(i in 1:length(possible_interventions)){
    intervention_indcs = which(interventions == possible_interventions[i])
    n_intervention = length(intervention_indcs)
    indcs = c(indcs, sample(intervention_indcs, n_intervention, replace=TRUE))
  }
  return(indcs)
}

run_gies = function(data, interventions){
  data = as.data.frame(data)
  p = ncol(data)
  colnames(data) = as.character(1:p)
  interventions = as.character(interventions)
  corr_mat = cor(data[interventions == '-1', ]) # -1 is flag for observational data
  all_targets = list()
  all_targets[[1]] = integer(0) # observation data marker
  possible_interventions = unique(interventions)
  possible_interventions = possible_interventions[possible_interventions != '-1']
  intervention_index_original = as.numeric(interventions)
  intervention_index = as.numeric(interventions)
  if(length(possible_interventions) > 0){
    for(i in 1:length(possible_interventions)){
      all_targets[[i + 1]] = as.numeric(possible_interventions[i])
    }
    for(i in 2:length(all_targets)){
      intervention_index[intervention_index_original == possible_interventions[i-1]] = i
    }
  }
  intervention_index[intervention_index == -1] = 1 # observational data is at index 1 in all_targets list
  gie_score_fn <- new("GaussL0penIntScore", data, all_targets, as.numeric(intervention_index)) # BIC score
  gies.fit <- gies(gie_score_fn)
  weights = gies.fit$repr$weight.mat()
  return(weights)
}
