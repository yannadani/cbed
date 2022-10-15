suppressMessages(library(pcalg))
suppressMessages(library(gRbase))

source('models/dag_bootstrap_lib/gies_helper.r') # ASSUMES RUN FROM NEW DIRECTORY

args = commandArgs(trailingOnly=TRUE)
n_boot = as.numeric(args[1])
data_path = args[2]
intervention_path = args[3]
seed = args[4]
path = args[5]

data = as.data.frame(read.table(data_path))
interventions = as.character(read.csv(intervention_path)[, 1])

boot_samps = bootstrap_gies(n_boot, data, interventions, seed, path)
