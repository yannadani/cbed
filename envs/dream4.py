import os
import uuid
from pathlib import Path
import glob
import shutil
import json
import numpy as np
import pathlib
import pandas as pd
from collections import namedtuple
import subprocess

from xml.dom import minidom

from .causal_environment import CausalEnvironment, Data
import networkx as nx
import graphical_models
import tqdm

from .utils import expm_np, num_mec


def create_tmp(base):
    uid = str(uuid.uuid4())
    tmp_path = Path(base) / uid
    os.makedirs(tmp_path, exist_ok=True)
    return tmp_path


def observe(samples, path, name='insilico_size10_1'):
    network = f'../configurations/{name}.xml'
    cmd = f'java -jar ../gnw-3.1b.jar -c ../settings.txt --input-net {network} --output-net-format=1 --simulate'

    data = []
    os.makedirs('envs/dream4/tmp', exist_ok=True)
    for _ in tqdm.tqdm(range(samples)):
        tmp_path = create_tmp('envs/dream4')
        subprocess.check_call(cmd.split(' '), cwd=tmp_path, stderr=subprocess.DEVNULL)
        data.append(pd.read_csv(f'{tmp_path}/{name}_wildtype.tsv', sep='\t'))
        shutil.rmtree(tmp_path)
    return pd.concat(data).to_numpy()


def intervene(node, path, name='insilico_size10_1'):
    os.makedirs('envs/dream4/tmp', exist_ok=True)
    network = f'../configurations/{name}.xml'
    cmd = f'java -jar ../gnw-3.1b.jar -c ../settings.txt --input-net {network} --output-net-format=1 --simulate'
    tmp_path = create_tmp('envs/dream4')
    subprocess.check_call(cmd.split(' '), cwd=tmp_path, stderr=subprocess.DEVNULL)
    data = pd.read_csv(f'{tmp_path}/{name}_knockouts.tsv', sep='\t')
    shutil.rmtree(tmp_path)
    return data.iloc[node].to_numpy()


def get_network(xml):
    xmldoc = minidom.parse(str(xml))

    nodes = []
    var2id = {}
    for i, node in enumerate(xmldoc.getElementsByTagName('species')):
        name = node.attributes.get('id').value
        if 'void' not in name:
            nodes.append(name)
            var2id[name] = i

    A = np.zeros((len(nodes), len(nodes)))

    for node in xmldoc.getElementsByTagName('reaction'):
        # child
        child = node.getElementsByTagName('listOfProducts')[0].getElementsByTagName('speciesReference')[0].attributes.get('species').value

        #parents
        for parent in node.getElementsByTagName('modifierSpeciesReference'):
            _from = var2id[parent.attributes.get('species').value]
            _to = var2id[child]
            A[_from, _to] = 1

    return nodes, var2id, A

class Dream4Environment(CausalEnvironment):

    def __init__(self, data_seed, path, name, logger):
        self.logger = logger

        self.path = path
        self.name = name
        self.nodes, self.var2id, A = get_network(pathlib.Path(path) / f'{name}.xml')
        self.adjacency_matrix = A
        self.num_nodes = len(self.nodes)

        self.id2var = {}
        for key, val in self.var2id.items():
            self.id2var[val] = key

        G = nx.convert_matrix.from_numpy_matrix(A, create_using=nx.DiGraph)
        self.graph = nx.relabel_nodes(G, self.id2var)

        self.xml_file = pathlib.Path(path) / f'{name}.xml'
        self.held_out_data = np.array([])

        if not os.path.exists('envs/dream4/gnw-3.1b.jar'):
            print('Downloading GeneWeaver')
            cmd = 'wget https://github.com/linlinchn/gnw/raw/master/gnw-3.1b.jar'
            subprocess.check_call(cmd.split(' '), cwd='envs/dream4/', stderr=subprocess.DEVNULL)

    def __getitem__(self, index):
        return self.samples[index]

    def dag(self):
        return graphical_models.DAG.from_nx(self.graph)

    def reseed(self, seed=None):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def intervene(self, iteration, num_samples, node, value_sampler):
        """Perform intervention to obtain a mutilated graph"""

        assert num_samples == 1, 'you cannot have more than one interventional sample per intervention for DREAM4'
        data = intervene(node, self.path, self.name)[None, ...]

        self.logger.log_interventions(iteration, [node]*num_samples, data[:, node])

        return Data(samples=data, intervention_node=node)

    def sample(self, num_samples):
        cache_file = pathlib.Path(self.path) / f'{self.name}_observations_{num_samples}.npz'

        if os.path.exists(cache_file):
            data = np.load(cache_file)['data']
        else:
            data = observe(num_samples, self.path, self.name)
            np.savez_compressed(cache_file, data=data)

        return Data(samples=data, intervention_node=-1)
