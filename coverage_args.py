import argparse
import datetime

import os

def get_args():
				parser = argparse.ArgumentParser()

				## Required parameters

				## imask parameters
				parser.add_argument('--bins-word', type=int, default=10, help='downsampling the vocabulary')
				parser.add_argument('--bins-attention', type=int, default=10, help='downsampling the vocabulary')
				parser.add_argument('--max-seq-len', type=int, default=128, help='downsampling the vocabulary')
				parser.add_argument('--batch-size', type=int, default=128, help='batch size for initialization')
				parser.add_argument('--threshold', type=float, default=0.0, help='batch size for initialization')
				parser.add_argument('--alpha', type=float, default=1.0, help='batch size for initialization')
				
				parser.add_argument('--word-importance-threshold', type=float, default=0.5, help='batch size for initialization')
				parser.add_argument('--interaction-importance-threshold', type=float, default=0.5, help='batch size for initialization')
				
				parser.add_argument('--suite', type = str, default = 'sentiment', help = 'which dataset to test')
				parser.add_argument('--type', type = str, default = 'inv', help = 'which dataset to test')
				parser.add_argument('--seed', type=int, default=10, help='downsampling the vocabulary')
				parser.add_argument('--save-dir', type = str, default = 'coverage_results/', help = 'which dataset to test')
				parser.add_argument('--base-model', type = str, default = 'bert-base-uncased', help = 'which dataset to test')
				parser.add_argument('--debug', type=int, default=0, help='batch size for initialization')
				#parser.add_argument('--baseline', action = 'store_true', help='batch size for initialization')
				#parser.add_argument('--specify-test', action = 'store_true', help='batch size for initialization')
				parser.add_argument('--test-name', type = str, default = '', help = 'which dataset to test')
				parser.add_argument('--query-tests-only', action = 'store_true', help='batch size for initialization')
				parser.add_argument('--shuffle', action = 'store_true', help='batch size for initialization')
				parser.add_argument('--init-point', action = 'store_true', help='batch size for initialization')
				parser.add_argument('--subset', type=int, default=1500, help='downsampling the vocabulary')
				parser.add_argument('--num_perturb', type=int, default=50, help='downsampling the vocabulary')
				parser.add_argument('--num-examples', type=int, default=867, help='downsampling the vocabulary')
				parser.add_argument('--sampling-ratio', type=int, default=2.0, help='batch size for initialization')
				parser.add_argument('--mask', default = '', type = str, help='batch size for initialization')
				parser.add_argument('--project', default = 'coverage for NLP', type = str, help='batch size for initialization')
				parser.add_argument('--notes', default = '', type = str, help='batch size for initialization')
				
				parser.add_argument('--root_file', default = './', type = str, help='batch size for initialization')
				
				args = parser.parse_args()
				
						

				
				
				return args
		
