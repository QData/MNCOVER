


import checklist
import copy
import torch
import random
import numpy as np
import pickle
import wandb
from textattack import metrics
from math import floor

from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from sst_model import *
from config import *

from textattack.coverage import neuronMultiSectionCoverage
from textattack.datasets import HuggingFaceDataset
from datasets import load_dataset
from coverage_args import *
from coverage_utils import *
from repr_utils import *





def load_model_dataset(args):

		if args.suite == 'sentiment':
				# pretrained BERT model on SST-2
				args.model_name_or_path = 'textattack/'+args.base_model+'-SST-2'
				print('='*5, 'Loading ', args.model_name_or_path, '='*5)
				print()
				print()
				model = SSTModel(args.model_name_or_path, args.max_seq_len)
		elif args.suite == 'qqp':
				# pretrained BERT model on QQP
				args.model_name_or_path = 'textattack/'+args.base_model+'-MRPC'
				print('='*5, 'Loading ', args.model_name_or_path, '='*5)
				print()
				print()
				model = SSTModel(args.model_name_or_path, args.max_seq_len)
		else:
				quit()
		trainset_str, validset_str = [], []
		if args.suite == 'sentiment':
				text_key = 'sentence'

				trainset = HuggingFaceDataset('glue', 'sst2', 'train', shuffle = True)
				validset = HuggingFaceDataset('glue', 'sst2', 'validation', shuffle = False)
				trainset_str = [example[0][text_key] for example in trainset]
				testset_str = [example[0][text_key] for example in validset]
				validset_str = trainset_str[floor(0.8*len(trainset_str)):]
				trainset_str = trainset_str[:floor(0.8*len(trainset_str))]
		if args.suite == 'qqp':
				
				'''
				trainset = load_dataset('quora')['train']['questions']

				print('total examples in quora dataset:', len(trainset))
				validset_str = trainset[floor(0.8*len(trainset)):]
				trainset_str = trainset[:floor(0.8*len(trainset))]
				print('Trainset in QQP dataset:', len(trainset))
				'''
				trainset = HuggingFaceDataset('glue', 'qqp', 'train', shuffle = True)
				validset = HuggingFaceDataset('glue', 'qqp', 'validation', shuffle = False)
				testset = HuggingFaceDataset('glue', 'qqp', 'test', shuffle = False)
				
				trainset_str = [(example[0]['question1'], example[0]['question2']) for example in trainset]
				validset_str = [(example[0]['question1'], example[0]['question2']) for example in validset]
				testset_str = [(example[0]['question1'], example[0]['question2']) for example in testset]
				
				

		
	

		
		
								
		if args.mask == 'imask':
				interaction_importance = torch.from_numpy( np.load('masks/'+args.suite+'_'+args.base_model + '-interaction.npy') )
				word_importance = torch.from_numpy( np.load('masks/'+args.suite+'_'+args.base_model + '-word.npy') ) 
				
				
				
				

				word_importance = torch.where(word_importance>=args.word_importance_threshold, torch.ones_like(word_importance),\
						 torch.zeros_like(word_importance))
				interaction_importance = torch.where(interaction_importance>=args.interaction_importance_threshold, torch.ones_like(interaction_importance),\
						 torch.zeros_like(interaction_importance))
			

						
		
		else:
				word_importance = None
				interaction_importance = None

				

		
		return model, trainset_str, validset_str, testset_str, word_importance, interaction_importance

