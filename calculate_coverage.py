'''
__author__ = Arshdeep Sekhon
'''
import checklist
import copy
import torch
import random
import numpy as np
import pickle
from textattack import metrics


from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from sst_model import *
from config import *

from textattack.coverage import neuronMultiSectionCoverage
from textattack.datasets import HuggingFaceDataset
from coverage_args import *
from coverage_utils import *
from repr_utils import *

from loading_saving_utils import *
from transformers import pipeline



args = get_args()
set_seed(args.seed)

# preloaded checklist suites
suite_path = 'tests/'+ SUITE_PATH[args.suite]
suite = TestSuite.from_file(suite_path)
model, trainset_str, validset_str, testset_str, word_importance, interaction_importance = load_model_dataset(args)
coverage = initialize_coverage_object(args, word_importance, interaction_importance)
print('initializing from training data!')

save_coverage_init_file = 'initialization/' + args.base_model + '_' + args.suite


if not os.path.exists(save_coverage_init_file+'.pkl'):
    print('can\'t find!: ', save_coverage_init_file+'.pkl')
    quit()
else:
    
    print('*'*100)
    print('exists!' , save_coverage_init_file+'.pkl')
    print('*'*100)
    coverage_loaded = pickle.load(open(save_coverage_init_file+'.pkl', 'rb'))
    coverage.coverage_attention_dicts = coverage_loaded.coverage_attention_dicts
    coverage.coverage_word_dicts = coverage_loaded.coverage_word_dicts
    coverage.min_word_coverage_tracker = coverage_loaded.min_word_coverage_tracker
    coverage.max_word_coverage_tracker = coverage_loaded.max_word_coverage_tracker
    coverage.min_attention_coverage_tracker = coverage_loaded.min_attention_coverage_tracker
    coverage.max_attention_coverage_tracker = coverage_loaded.max_attention_coverage_tracker
    del coverage_loaded
    initial_coverage = coverage._compute_coverage() 


test = suite.tests[args.test_name]
input_examples_orig = suite.tests[args.test_name].data
input_examples = []


test_examples = (suite.tests[args.test_name].to_raw_examples()) 
shuffled_indices = list(range(len(test_examples)))[0:args.subset]

test_examples = [test_examples[t] for t in shuffled_indices]
test_indices = [suite.tests[args.test_name].example_list_and_indices()[1][t] for t in shuffled_indices] # correspond to which original seed
        
for initial, final in zip(test_indices, test_examples):
    if type(input_examples_orig[initial]) is not list:
        input_examples_orig[initial] = [input_examples_orig[initial]]
    input_examples.append(input_examples_orig[initial][0])

if hasattr(suite.tests[args.test_name], 'labels'):
    labels = suite.tests[args.test_name].labels
    if TYPE_MAP[type(suite.tests[args.test_name])] == 'MFT' and type(suite.tests[args.test_name].labels) is not list:
        labels = [labels]*len(test_examples)



# coverage filtering 
original_number_test_examples = len(test_examples)
relevant_idxs, test_examples_list, skipped_examples_list, selected_tracker, coverage_values = filter_using_coverage(coverage, initial_coverage, test_examples, args.threshold)
new_tests = (([test_indices[d] for d in relevant_idxs]))



result_dict = pickle.load(open('baseline_results/sentiment.pkl','rb'))[args.base_model.split('-')[0]][args.test_name]



print('calculating failure rate!')
coverage = coverage_values[selected_tracker == 1]
ids = relevant_idxs[selected_tracker == 1]



cov_logger = {}
cov_logger[-1] = {'prev_id': None, 'cov':[0.0]}
prev_id = 0
prev_prev_id = -1
index= 0
while index<len(ids):
		id = ids[index]
		if id not in cov_logger.keys():
				cov_logger[id] = {'prev_id': prev_prev_id, 'cov':[]}
		cov_logger[id]['cov'].append(coverage[index])
		if id!=prev_id:
				prev_prev_id = id
		else:
				prev_id = id
		index+=1
cov_logger[ids[1]]['prev_id'] = ids[0]
#print(cov_logger)
cov_delta = {}
for id in cov_logger.keys():
		if id!=-1:
				cov_delta[id] = cov_logger[id]['cov'][-1] - cov_logger[cov_logger[id]['prev_id']]['cov'][-1]


#print(result_dict.keys())
test = args.test_name
fail_rates = []
lens = []
for t in np.quantile([cov_delta[key] for key in cov_delta], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):

		new_ids = [key for key in cov_delta if cov_delta[key]>=t]
		if len(new_ids)>0:
				#print(suite.tests[test].data)
				n_run = n = len(list(set([idx for idx in suite.tests[args.test_name].example_list_and_indices()[1] if idx in new_ids])))
				if suite.tests[test].run_idxs is not None:
								n_run = len(set(([idx for idx in result_dict['run_idxs'] if idx in new_ids])))
				
				
				orig_fails = len(result_dict['fails'])
				fails = len([idx for idx in result_dict['fails'] if idx in new_ids])
				
				
				fail_rate = 100 * fails / len(new_ids)
				orig_fail_rate = 100 * orig_fails / len(set(result_dict['run_idxs']))
				
				

				print(args.test_name, t, fail_rate)
				fail_rates.append(fail_rate)
				lens.append(len(new_ids))




orig_lens = len(set([idx for idx in suite.tests[args.test_name].example_list_and_indices()[1]]))




print(max(fail_rates), sum(fail_rates)/len(fail_rates))			




