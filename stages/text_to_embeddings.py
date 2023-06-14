import numpy as np
import pandas as pd

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter
import torch

import tempfile
import datetime
import time
import os
from pathlib import Path
import subprocess
import re
import shutil
import importlib
import random
import json
import os

import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import sentence_transformers
from sentence_transformers import SentenceTransformer, util

import psutil

import umap
import matplotlib.pyplot as plt
import seaborn as sns


class Embeddings:
    """Class for creating, storing, and visualizing embeddings."""

    
    def __init__(self, model_name=None, vec=None, 
                 make_model=AutoModel.from_pretrained, 
                 make_tok=AutoTokenizer.from_pretrained):
        """
        Initialize embeddings object with model, tokenizer, and, if available,
        pre-made vectors of embeddings.
        :param model: BERT model instance.
        :param tokenizer: BERT tokenizer instance.
        :param vec: Optional vectors for embeddings.
        """

        # model paths lookup table
        self.model_paths = {
            'sbert_mean':               'sentence-transformers/bert-base-nli-mean-tokens',
            'sbert_max':                'sentence-transformers/bert-base-nli-max-tokens',
            'sbert_cls':                'sentence-transformers/bert-base-nli-cls-token',
            'sdistilbert_mean':         'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            'sroberta_mean':            'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            'bert':                     'bert-base-uncased',
            'bart_mnli':                'facebook/bart-large-mnli',
            'roberta_news':             'allenai/news_roberta_base',
            'distilbert_mnli':          'huggingface/distilbert-base-uncased-finetuned-mnli',
            'roberta':                  'roberta-base',
            'distilbert_sst':           'distilbert-base-uncased-finetuned-sst-2-english',
            'albert':                   'albert-base-v2',
            'roberta_sst':              'textattack/roberta-base-SST-2',
            'albert_sst':               'textattack/albert-base-v2-SST-2',
            'hyperpart_tapt_515':       'allenai/dsp_roberta_base_tapt_hyperpartisan_news_515',
            'hyperpart_dapt_tapt_5015': 'allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_5015',
            'hyperpart_tapt_5015':      'allenai/dsp_roberta_base_tapt_hyperpartisan_news_5015',
            'hyperpart_dapt_tapt_515':  'allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_515',
            'longformer-base':          'allenai/longformer-base-4096'
        }
        
        # assert valid model provided
        assert model_name is not None, "No model provided!"
        assert model_name in self.model_paths, "Not a valid model!"

        # init
        self.model = make_model(self.model_paths[model_name], return_dict=True, output_hidden_states=True)
        self.tokenizer = make_tok(self.model_paths[model_name])
        self.vec = {}
        

    def transformers_embeddings(self, sentences, model, tokenizer, layer=-2, pooling='mean', max_len=None):
        """
        Create BERT sentence embeddings.
        :param sentences: List of sentences to be embedded.
        :param model: BERT model instance.
        :param tokenizer: BERT tokenizer instance.
        :param layer: Pooling/aggregation layer for constructing sentence
        level embeddings (default: second-to-last).
        :return: Tensor of sentence embeddings.
        """

        if isinstance(self.model, transformers.LongformerModel):
            
            print("Using Longformer... setting global attention to and using <cls> token...")
            # Longformer specific settings
            # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
            inputs = tokenizer(sentences, padding=True, truncation=True, max_length=4096)
            input_ids = torch.tensor(inputs['input_ids'])
            attention_mask = torch.tensor(inputs['attention_mask'])
            global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
            global_attention_mask[:, [0]] = 1 
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, 
                                global_attention_mask=global_attention_mask)
            hidden_states = outputs.hidden_states
            token_vecs = hidden_states[layer]
            sentence_embedding = token_vecs[:, 0, :]
            return sentence_embedding
            
        else:
            
            # run transformer on sentence
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, 
                               max_length=max_len, )
            with torch.no_grad():
                outputs = model(**inputs)

            # extract second to last hidden state
            hidden_states = outputs.hidden_states
            token_vecs = hidden_states[layer]
            att_mask = inputs['attention_mask']

            # compute sentence embedding via mean
            if pooling=='mean':
                sentence_embedding = self.mean_pooling(token_vecs, att_mask)

            elif pooling=='cls':
                sentence_embedding = token_vecs[:, 0, :]

            elif pooling=='cat':
                sentence_embedding = torch.cat((hidden_states[-1][:, 0, :], hidden_states[-2][:, 0, :],
                                                hidden_states[-3][:, 0, :], hidden_states[-4][:, 0, :]), dim=1)

            elif pooling=='max':
                sentence_embedding = self.max_pooling(token_vecs, att_mask)

            else:
                sentence_embedding = self.mean_pooling(token_vecs, att_mask)

            return sentence_embedding

        
    def embed(self, name=None, data=None, **kwargs):
        """
        Embed data, e.g., sentences, headlines.
        :param data: List of strings, e.g., sentences, headlines, to be embedded.
        """
        
        self.vec[name] = self.transformers_embeddings(data, self.model, self.tokenizer, **kwargs)
                 

    def start_visualization(self, name, labels, port=6006, data_dir="embeddings_projector_data"):
        """
        Create server and write data for visualization.
        :param labels: List of strings that act as labels for data points (to aid with visualization).
        :param port: Integer for port of server; 6006 is Tensorboard default.
        :param data_dir: Local directory path for writing embedding data read by Tensorboard server.
        """
        
        # assert valid tensor
        assert len(self.vec) > 0, "No embedding vectors/tensor found!"
        
        # save data directory
        self.data_dir = data_dir
        
        # write embeddings to projector and call tensorboard projector
        with SummaryWriter(log_dir=data_dir) as writer:
            writer.add_embedding(self.vec[name], metadata=labels)
            # %reload_ext tensorboard
            # %tensorboard --logdir={data_dir} --port={port} --host localhost
            
    
    def end_visualization(self):
        """
        Clean up servers and directories created by Tensorboard visualizations.
        """
        
        # kill all servers created by tensorboard
        for info in tb.notebook.manager.get_all():
            try:
                os.kill(info.pid, signal.SIGTERM)
                print(f"Process with PID={info.pid} has been terminated.") 
            except ProcessLookupError:
                pass
        
        # clean up data directory for embedding projection
        try:
            shutil.rmtree(self.data_dir)
            print(f"Embedding projector directory '{self.data_dir}' has been deleted.") 
        except FileNotFoundError:
            pass
        
    
    def umap_dim_reduc(self, name, **kwargs):
        """
        Create embeddings with reduced dimensionality (2D) using UMAP.
        """
        
        data_array = self.vec[name].detach().numpy()
        reducer = umap.UMAP(**kwargs)
        embedding = reducer.fit_transform(data_array)
        
        self.reduc_tensor = embedding
        
        return embedding
    

    def umap_plot(self, names, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title='', color=None, alpha=1.0):
        
        # source: https://umap-learn.readthedocs.io/en/latest/parameters.html
        data_array = self.vec[names[0]].detach().numpy()
        for name in names[1:]:
            data_array = np.concatenate((data_array, self.vec[name].detach().numpy()), axis=0)
        
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        
        u = fit.fit_transform(data_array)
        fig = plt.figure()
        
        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(u[:,0], range(len(u)), c=color, alpha=alpha)
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(u[:,0], u[:,1], c=color, alpha=alpha)
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(u[:,0], u[:,1], u[:,2], c=color, alpha=alpha, s=100)
            
        plt.title(title, fontsize=18)
        plt.show()


    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        # adapted from https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    
    @staticmethod
    def max_pooling(token_embeddings, attention_mask):
        """Max Pooling - Take the max value over time for every dimension"""
        # adapted from https://huggingface.co/sentence-transformers/bert-base-nli-max-tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_over_time = torch.max(token_embeddings, 1)[0]
        return max_over_time