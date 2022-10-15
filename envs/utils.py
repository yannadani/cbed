import graphical_models
import numpy as np


def matrix_poly_np(matrix, d):
	x = np.eye(d) + matrix/d
	return np.linalg.matrix_power(x, d)

def expm_np(A, m):
	expm_A = matrix_poly_np(A, m)
	h_A = np.trace(expm_A) - m
	return h_A

def num_mec(m):
	a = graphical_models.DAG.from_nx(m)
	skeleton = a.cpdag() ##Find the skeleton
	all_dags = skeleton.all_dags() #Find all DAGs in MEC
	return len(all_dags)