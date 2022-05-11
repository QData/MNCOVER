import copy
import numpy as np
import torch 
import wandb
import os
import textattack
#from textattack.coverage import neuronMultiSectionCoverage
from coverage_class import neuronMultiSectionCoverage
def get_init_filepaths(args, trainset_size):
		return os.path.join( args.root_file, args.base_model +'_TRAIN_INIT_'+str(args.init_point) + '_' + args.suite+'_BW_'+ str(args.bins_word) + \
		'_BA_' + str(args.bins_attention)	+ '_INIT_' + str(trainset_size))

def initialize_coverage_object(args, word_importance, interaction_importance):
		if args.base_model == 'bert-base-uncased' :
			pad = 0
		else:
			pad = 1
		return neuronMultiSectionCoverage(test_model = args.model_name_or_path, max_seq_len = args.max_seq_len, alpha = args.alpha, pad = pad,
																		bins_word = args.bins_word, bins_attention = args.bins_attention, bz = args.batch_size,
																		pre_limits = False, mask = (args.mask != ''), word_importance=word_importance, interaction_importance=interaction_importance)
		

def filter_using_coverage(coverage, initial_coverage, test_examples, threshold = 0.001):
						relevant_idxs = []
						coverage_list = []
						selected_tracker = []
						local_coverage = copy.deepcopy(coverage)
						initial_local_coverage = initial_coverage
						skipped_examples_list, test_examples_list = [],[]
						for idx, example in enumerate(test_examples):
								coverage_temp = copy.deepcopy(local_coverage)
								temp_coverage_value = coverage_temp([example] )
								print('CHANGE IN COVERAGE:', temp_coverage_value - initial_local_coverage)
								assert temp_coverage_value >= initial_local_coverage
								if (temp_coverage_value - initial_local_coverage) > threshold :
										print('Increases Coverage!', example)
										initial_local_coverage = temp_coverage_value
										del local_coverage
										local_coverage = copy.deepcopy(coverage_temp)
										del coverage_temp
										
										test_examples_list.append([idx, example])
										relevant_idxs.append(idx)
										coverage_list.append(temp_coverage_value)
										selected_tracker.append(1)
										wandb.log({'coverage': coverage_list[-1]})
								else:
										print('pass: ', example)
										coverage_list.append(temp_coverage_value)
										selected_tracker.append(0)
										skipped_examples_list.append([idx, example])

						return relevant_idxs, test_examples_list, skipped_examples_list, selected_tracker, coverage_list
def get_predictions_after_tx(sst_model, input_examples, test_examples):
						with torch.no_grad():
								predictions_before_tx = []
								predictions_prob_before_tx = []
								for example in input_examples:
										prob, label = sst_model(example)
										predictions_before_tx.append(label.item())
										predictions_prob_before_tx.append(prob[0,label.item()].item())
				
								predictions_after_tx = []
								predictions_prob_after_tx = []
								for example in test_examples:
										prob, label = sst_model(example)
										
										predictions_after_tx.append(label.item())
										predictions_prob_after_tx.append(prob[0,label.item()].item())
						return predictions_before_tx, predictions_prob_before_tx, predictions_after_tx, predictions_prob_after_tx
			 
