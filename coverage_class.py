import logging

import torch
import transformers
from tqdm import tqdm
import itertools
import copy
import numpy as np
import textattack
from collections import defaultdict
#from .coverage import ExtrinsicCoverage
import torch.nn.functional as F
import time
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


COVERAGE_MODEL_TYPES = ["bert", "albert", "distilbert", "roberta"]


class neuronMultiSectionCoverage():
				"""
				``neuronMultiSectionCoverage`` measures the neuron coverage acheived by a testset
				Args:
								test_model(Union[str, torch.nn.Module]): name of the pretrained language model from `transformers`
												or the actual test model as a `torch.nn.Module` class. Default is "bert base uncased" from `transformers`.
								tokenizer (:obj:``, optional): If `test_model` is not a pretrained model from `transformers, need to provide
												the tokenizer here.
								max_seq_len (int): Maximum sequence length accepted by the model to be tested.	However, if you are using a pretrained model from `transformers`, this is handled
												automatically using information from `model.config`.
								threshold(float): threshold for marking a neuron as activated
								coverage(str):	measure type of neuron coverage at the level of layer outputs
				"""

				def __init__(
								self,
								test_model="textattack/bert-base-uncased-ag-news",
								tokenizer=None,
								max_seq_len=-1,
								threshold=0.0,
								num_labels = 2,
								coverage = 'multisection', 
								pre_limits = False,
								bins_attention =4,
								bins_word = 4, 
								min_value=np.inf,
								pad = 0, 
								max_value=-np.inf,
								bz	= 128, 
								layers=12,
								mask = False,
								heads =12,
								word_importance = None, 
								interaction_importance = None,
								hidden = 768,
								alpha = 1.0,
				):

								self.coverage = coverage
								self.alpha = alpha
								self.mask = mask
								self.pre_limits = pre_limits
								self.bins_attention = bins_attention
								self.bins_word = bins_word	# number of sections for each neuron
								self.max_seq_len = 128
								self.model_type = 'bert'
								self.word_importance = word_importance
								self.hidden = hidden
								self.interaction_importance = interaction_importance
								self.PAD = pad
								
								config = transformers.AutoConfig.from_pretrained(
												test_model, output_hidden_states=True, num_labels = num_labels
								)
								if config.model_type in COVERAGE_MODEL_TYPES:
												self.test_model = (
																transformers.AutoModelForSequenceClassification.from_pretrained(
																				test_model, config=config
																)
												)
												self.test_model.tokenizer = transformers.AutoTokenizer.from_pretrained(
																test_model
												)
												self.model_type = self.test_model.config.model_type
												self.max_seq_len = (
																max_seq_len
																if max_seq_len != -1
																else self.test_model.config.max_position_embeddings
												)
								else:
												raise ValueError(
																"`neuronCoverage` only accepts models in "
																+ ",".join(COVERAGE_MODEL_TYPES)
												)
								
								
								self.test_model.to(textattack.shared.utils.device)
								self.threshold = threshold
								self.test_model.eval()

								# initialize min and max for coverage
								min_attention_value = min_value
								max_attention_value = max_value
								if pre_limits:
										min_attention_value = 0.0
										max_attention_value = 1.0

								self.coverage_output_dicts = torch.zeros((self.bins_word+3, num_labels))
								self.coverage_word_dicts = torch.zeros((self.bins_word+3, layers+1, self.max_seq_len, self.hidden))
								self.coverage_attention_dicts = torch.zeros((self.bins_attention + 3, layers, heads, self.max_seq_len, self.max_seq_len))
								self.min_word_coverage_tracker = torch.zeros((layers+1, self.max_seq_len, self.hidden)).fill_(min_value)
								self.min_attention_coverage_tracker = torch.zeros((layers, heads, self.max_seq_len, self.max_seq_len)).fill_(min_attention_value)

								self.max_word_coverage_tracker = torch.zeros(( layers+1, self.max_seq_len, self.hidden)).fill_(max_value)
								self.max_attention_coverage_tracker = torch.zeros(( layers, heads, self.max_seq_len, self.max_seq_len)).fill_(max_attention_value)
								self.max_output_coverage_tracker = torch.zeros((num_labels)).fill_(max_value)
								self.min_output_coverage_tracker = torch.zeros((num_labels)).fill_(min_value)
								
								
		
								if 'snac' in self.coverage:
										self.k_m = 2
								if 'nbc' in self.coverage:
										self.k_m = 1
								'''
								for i in range(self.bins_word):
												word_tracker = self._init_word_coverage(fill_value=0.0)
												self.coverage_word_dicts.append(word_tracker)
								for i in range(self.bins_attention):
												attention_tracker = self._init_attention_coverage(fill_value=0.0)
												self.coverage_attention_dicts.append(attention_tracker)
								'''
				def _init_word_coverage(self, fill_value):
								"""Initialize `coverage_tracker` dictionary

								Returns:
												`coverage_tracker`(dict): a dictionary with key: neuron and value: (bool) intialized False
								"""
								coverage_word_tracker = torch.zeros_like(self.coverage_word_dicts)
				def _init_output_coverage(self, fill_value):
								"""Initialize `coverage_tracker` dictionary

								Returns:
												`coverage_tracker`(dict): a dictionary with key: neuron and value: (bool) intialized False
								"""
								coverage_output_tracker = torch.zeros_like(self.coverage_output_dicts)
								
								

								return coverage_output_tracker
				def _init_attention_coverage(self, fill_value):
								"""Initialize `coverage_tracker` dictionary

								Returns:
												`coverage_tracker`(dict): a dictionary with key: neuron and value: (bool) intialized False
								"""
								# attention neurons
								coverage_attention_tracker = torch.zeros_like(self.coverage_attention_dicts)
								return coverage_attention_tracker

				def _update_output_layer_coverage(self, outputs):
							 self.max_output_coverage_tracker = torch.where(torch.max(outputs, dim = 0).values.detach() > self.max_output_coverage_tracker , torch.max(output, dim = 0).values.detach(), self.max_output_coverage_tracker)
							 self.min_output_coverage_tracker = torch.where(torch.min(outputs, dim = 0).values.detach() \
								 < self.min_output_coverage_tracker , torch.min(outputs, dim = 0).values.detach(), self.min_output_coverage_tracker)
								


				def _update_initial_word_coverage(self, embeddings):
								"""Update `coverage_tracker` for input `text` for coarse coverage
												Args:
								`text`(str): text to update neuron coverage of.

								"""

								'''
								encodings = self.test_model.tokenizer(text, return_tensors="pt")
								if self.max_seq_len > 0:
												input_ids = encodings.input_ids[:, : self.max_seq_len]
												attention_mask = encodings.attention_mask[:, : self.max_seq_len]

								input_ids = input_ids.to(textattack.shared.utils.device)
								attention_mask = attention_mask.to(textattack.shared.utils.device)
								outputs = self.test_model(input_ids, attention_mask=attention_mask)
								outputs[1][0]
								'''
								
								sentence_length = embeddings[0][0, ...].size(0)

								embeddings = [e.unsqueeze(1) for e in embeddings]
								
								embeddings = torch.cat(embeddings, dim = 1).cpu()
								
								
								#print(embeddings,, self.max_word_coverage_tracker.device)
								print(embeddings.size(),torch.max(embeddings, dim = 0).values.detach().size(), self.max_word_coverage_tracker.size())
								self.max_word_coverage_tracker = torch.where(torch.max(embeddings, dim = 0).values.detach() > self.max_word_coverage_tracker , torch.max(embeddings, dim = 0).values.detach(), self.max_word_coverage_tracker)
								self.min_word_coverage_tracker = torch.where(torch.min(embeddings, dim = 0).values.detach() \
								 < self.min_word_coverage_tracker , torch.min(embeddings, dim = 0).values.detach(), self.min_word_coverage_tracker)
								
								
								'''
								self.max_coverage_tracker["classifier"] = torch.where(
												(outputs[0][0, ...].detach()) > self.max_coverage_tracker["classifier"],
												outputs[0][0, ...].detach(),
												self.max_coverage_tracker["classifier"],
								)
								'''
								

				def _update_initial_attention_coverage(self, all_attentions):
								"""Update `coverage_tracker` for input `text` for coarse coverage
												Args:
								`text`(str): text to update neuron coverage of.

								"""

								
								# all_attentions	= list of attentions of size B X H X L X L 
								
								sentence_length = all_attentions[0][0,0, ...].size(-1)
								all_attentions = torch.cat([a.unsqueeze(1) for a in all_attentions], dim = 1) # B X LA X HD X L X L
								all_attentions_max = torch.max( all_attentions, dim = 0).values.cpu()
								all_attentions_min = torch.min( all_attentions, dim = 0).values.cpu()
								self.max_attention_coverage_tracker = torch.where(all_attentions_max > self.max_attention_coverage_tracker, all_attentions_max, self.max_attention_coverage_tracker)
								self.min_attention_coverage_tracker = torch.where(all_attentions_min < self.min_attention_coverage_tracker, all_attentions_min, self.min_attention_coverage_tracker)
								
				def _update_initial_output_coverage(self, output) :
								"""Update 
								"""
								min_outputs = torch.min(output, dim = 0).values.cpu()
								max_outputs = torch.max(output, dim = 0).values.cpu()
								self.max_output_coverage_tracker = torch.where(max_outputs > self.max_output_coverage_tracker, max_outputs, self.max_output_coverage_tracker)
								self.min_output_coverage_tracker = torch.where(min_outputs < self.min_output_coverage_tracker, min_outputs, self.min_output_coverage_tracker)


								
				def _update_initial_coverage(self, output, all_hidden_states, all_attentions, word_mask = None):
								"""Update `coverage_tracker` for input `text`
												Args:
								`text`(str): text to update neuron coverage of.

								"""
								
								
								self._update_initial_word_coverage(all_hidden_states)
								self._update_initial_output_coverage(output)
								self._update_initial_attention_coverage(all_attentions)
												
				def initialize_from_training_dataset(self, trainset, trainset2 = None,	bz=1):
								"""Update coverage from training dataset
								`trainset`(list[str]): training dataset coverage statistics 


								"""
								mask_no = 0
								
								
								start = 0
								with torch.no_grad():
										for t in tqdm(trainset):
												
												if mask_no + bz >= len(trainset):
														end = len(trainset)
												else:
														end = start + bz
												if start >=	end or start >= len(trainset) : break
												#print('current indices : ', trainset[start:end], start, end, len(trainset))
												#tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
												#													 max_length=max_length,
												#														return_token_type_ids=True, truncation=True)
												if hasattr(self.test_model.tokenizer, "batch_encode_plus"):
														if trainset2 is None:

																if isinstance(trainset[start:end], tuple) and len(trainset[start:end]) == 1:
																 # Unroll tuples of length 1.
																	 input_text_list = [t[0] for t in trainset[start:end]]
																encodings = self.test_model.tokenizer.batch_encode_plus(
																								trainset[start:end],
																								truncation=True,
																								return_token_type_ids=True, 
																								max_length=self.max_seq_len,
																								add_special_tokens=True,
																								padding="max_length",
																						)
												

																#print([v for k,v in encodings.data.items()])
																encodings = {k: torch.cat([torch.LongTensor(c).unsqueeze(0).to(textattack.shared.utils.device) for c in v], dim = 0) for k, v in encodings.data.items()}
														else:
																
																				#input_text_list = [t[0] for t in trainset[start:end]]
																				encodings = self.test_model.tokenizer.batch_encode_plus(
																								trainset[start:end],trainset2[start:end],
																								truncation=True,
																								return_token_type_ids=True, 
																								max_length=self.max_seq_len,
																								padding="max_length",
																						)
												

																				#print([v for k,v in encodings.data.items()])
																				encodings = {k: torch.cat([torch.LongTensor(c).unsqueeze(0).to(textattack.shared.utils.device) for c in v], dim = 0) for k, v in encodings.data.items()}
																
												else:
														def encode(input_text):
																if isinstance(input_text, str):
																		input_text = (input_text,)
																encoded_text = self.tokenizer.encode_plus(
																										*input_text,
																										max_length=self.max_length,
																										add_special_tokens=True,
																										padding="max_length",
																										truncation=True,
																								)
																return dict(encoded_text)
														encodings = [encode(input_text) for input_text in trainset[start:end]]
												#encodings = self.test_model.tokenizer(trainset[start:end], padding='max_length', truncation=True, return_tensors="pt", max_length = self.max_seq_len)
												#print(encodings)

												
												'''
												if self.max_seq_len > 0:
														input_ids = encodings['input_ids'][:, : self.max_seq_len]
														attention_mask = encodings['attention_mask'][:, : self.max_seq_len]
														token_type_ids = encodings['token_type_ids'][:, : self.max_seq_len]
												
												input_ids = input_ids.to(textattack.shared.utils.device)
												attention_mask = attention_mask.to(textattack.shared.utils.device)
												token_type_ids = token_type_ids.to(textattack.shared.utils.device)
												'''
												outputs = self.test_model(**encodings, output_attentions=True,output_hidden_states=True)
												
												#outputs = self.test_model(input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states = True, token_type_ids = token_type_ids)
												all_hidden_states, all_attentions = outputs[-2:]
												
												self._update_initial_coverage(outputs[0], all_hidden_states, all_attentions)
												start = end


								# self.training_word_coverage_dicts = copy.deepcopy(self.coverage_word_dicts)
								# self.training_attention_coverage_dicts = copy.deepcopy(self.coverage_attention_dicts)

				def _eval(self, text):
								"""Update `coverage_tracker` for input `text` for coarse coverage
												Args:
								`text`(str): text to update neuron coverage of.

								"""
								encodings = self.test_model.tokenizer(text, return_tensors="pt")
								if self.max_seq_len > 0:
												input_ids = encodings.input_ids[:, : self.max_seq_len]
												attention_mask = encodings.attention_mask[:, : self.max_seq_len]
												token_type_ids = encodings.token_type_ids[:, : self.max_seq_len]
								input_ids = input_ids.to(textattack.shared.utils.device)
								attention_mask = attention_mask.to(textattack.shared.utils.device)
								outputs = self.test_model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
								return outputs
				def _update_output_coverage(self, outputs):
								
								"""Update `coverage_tracker` for input `text` for coarse coverage
												Args:
								`text`(str): text to update neuron coverage of.

								
								
								a = time.time()
								encodings = self.test_model.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length = self.max_seq_len)
								if self.max_seq_len > 0:
												input_ids = encodings.input_ids[:, : self.max_seq_len]
												attention_mask = encodings.attention_mask[:, : self.max_seq_len]

								input_ids = input_ids.to(textattack.shared.utils.device)
								attention_mask = attention_mask.to(textattack.shared.utils.device)
								outputs = self.test_model(input_ids, attention_mask=attention_mask)
								b = time.time()
								
								sentence_length = outputs[1][0][0, ...].size(0)
								"""
								#print('size of output hidden bectors: ', hidden_vectors.size())
								
								current_coverage_tracker = self._init_output_coverage(fill_value=0)
								a = time.time()
								section_length = (self.max_output_coverage_tracker - self.min_output_coverage_tracker ) / self.bins_word
								section_length = section_length.unsqueeze(0).repeat(outputs.size(0), 1)
								#print('section length: ', section_length.size())
								section_index = torch.where(
												section_length > 0,
												(
																torch.floor(
																				(
																								outputs.cpu().detach()
																								- self.min_output_coverage_tracker
																				)
																				/ section_length
																)
												),
												torch.zeros_like(outputs.cpu().detach(), requires_grad=False) -1,
								).long()
								# print('section index: ', section_index.size())
								
								
								#section_index = torch.where(section_index, section_index, self.bins_word + 1)
								#section_index = torch.where(section_index>0, section_index, torch.zeros_like(section_index) + self.bins_word + 1)
								section_index = torch.where(section_index<self.bins_word, section_index, torch.zeros_like(section_index) + self.bins_word + 2)
								section_index = torch.where(section_index>0, section_index, torch.zeros_like(section_index) + self.bins_word + 1)
								
								# print('section index: ', section_index.size())

								temp_store_activations =	(F.one_hot(section_index, num_classes = self.bins_word + 3)).permute(0,2,1)
								temp_store_activations = torch.max(temp_store_activations, dim = 0).values
								# print('Temp Store Activations: ', temp_store_activations.size())
								self.coverage_output_dicts += temp_store_activations
								del temp_store_activations
								
								del current_coverage_tracker

				
				def _update_word_coverage(self, all_hidden_states, word_mask = None):
								
								"""Update `coverage_tracker` for input `text` for coarse coverage
												Args:
								`text`(str): text to update neuron coverage of.

								
								
								a = time.time()
								encodings = self.test_model.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length = self.max_seq_len)
								if self.max_seq_len > 0:
												input_ids = encodings.input_ids[:, : self.max_seq_len]
												attention_mask = encodings.attention_mask[:, : self.max_seq_len]

								input_ids = input_ids.to(textattack.shared.utils.device)
								attention_mask = attention_mask.to(textattack.shared.utils.device)
								outputs = self.test_model(input_ids, attention_mask=attention_mask)
								b = time.time()
								
								sentence_length = outputs[1][0][0, ...].size(0)
								"""
								hidden_vectors = torch.cat([o.unsqueeze(1) for o in all_hidden_states], dim = 1)
								sentence_length = hidden_vectors.size(2)
								#print('size of output hidden bectors: ', hidden_vectors.size())
								current_coverage_tracker = self._init_word_coverage(fill_value=0)
								a = time.time()
								section_length = (self.max_word_coverage_tracker - self.min_word_coverage_tracker ) / self.bins_word
								section_length = section_length.unsqueeze(0).repeat(hidden_vectors.size(0), 1, 1, 1)
								#print('section length: ', section_length.size())
								section_index = torch.where(
												section_length > 0,
												(
																torch.floor(
																				(
																								hidden_vectors.cpu().detach()
																								- self.min_word_coverage_tracker
																				)
																				/ section_length
																)
												),
												torch.zeros_like(hidden_vectors.cpu().detach(), requires_grad=False) -1,
								).long()
								# print('section index: ', section_index.size())
								
								
								#section_index = torch.where(section_index, section_index, self.bins_word + 1)
								#section_index = torch.where(section_index>0, section_index, torch.zeros_like(section_index) + self.bins_word + 1)
								section_index = torch.where(section_index<self.bins_word, section_index, torch.zeros_like(section_index) + self.bins_word + 2)
								section_index = torch.where(section_index>0, section_index, torch.zeros_like(section_index) + self.bins_word + 1)
								
								# print('section index: ', section_index.size())

								temp_store_activations =	(F.one_hot(section_index, num_classes = self.bins_word + 3)).permute(0,4,1,2,3)
								if self.mask :
										# temp mask is of size	B X 1 X 1 X length X 1
										# inside is B X bins X layers X length X hidden 
										temp_mask = (word_mask).unsqueeze(2).unsqueeze(1).unsqueeze(1).to(temp_store_activations.device)
										#temp_mask = temp_mask.repeat(1,temp_store_activations.size(0), temp_store_activations.size(1), 1, temp_store_activations.size(3))
										temp_store_activations = temp_store_activations*temp_mask
										del temp_mask
								temp_store_activations = torch.max(temp_store_activations, dim = 0).values
								# print('Temp Store Activations: ', temp_store_activations.size())
								self.coverage_word_dicts += temp_store_activations
								del temp_store_activations
								
								del current_coverage_tracker

				def _update_attention_coverage(self, all_attentions, attention_mask = None):
								"""Update `coverage_tracker` for input `text` for coarse coverage
												Args:
								`text`(str): text to update neuron coverage of.

								
								encodings = self.test_model.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length = self.max_seq_len)
								if self.max_seq_len > 0:
												input_ids = encodings.input_ids[:, : self.max_seq_len]
												attention_mask = encodings.attention_mask[:, : self.max_seq_len]

								input_ids = input_ids.to(textattack.shared.utils.device)
								attention_mask = attention_mask.to(textattack.shared.utils.device)
								outputs = self.test_model(input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states = True)
								
								all_hidden_states, all_attentions = outputs[-2:]
								# all_attentions	= list of attentions of size B X H X L X L 

								"""
								sentence_length = all_attentions[0][0,0, ...].size(-1)

								
								all_attentions = torch.cat( [a.unsqueeze(1) for a in all_attentions] , dim = 1).cpu()[:,:, 0:sentence_length, 0:sentence_length]
								# B X layers X heads X l X l
								# print('attentions size: ', all_attentions.size())
								current_coverage_tracker = self._init_attention_coverage(fill_value=0)
								

								section_length = (self.max_attention_coverage_tracker[:,:, 0:sentence_length, 0:sentence_length] - \
																self.min_attention_coverage_tracker[:,:, 0:sentence_length, 0:sentence_length] ) / self.bins_attention
								section_length = section_length.unsqueeze(0).repeat(all_attentions.size(0), 1, 1, 1, 1)
								# print(' section length: ', section_length.size())
								section_index = torch.where(
																section_length > 0,
																(
																		 torch.floor(
																				 (
																						 all_attentions.cpu().detach()
																						 - self.min_attention_coverage_tracker
																						 )
																				 /	section_length
																				 )
																		 ),
																torch.zeros_like(all_attentions.cpu().detach(), requires_grad=False) - 1
																).long()

								# print('section index: ', section_index.size())
								
								section_index = torch.where(section_index<self.bins_attention, section_index, torch.zeros_like(section_index) + self.bins_attention + 2)
								#print(section_index.max(), section_index.min(), self.bins_attention + 3, section_index.size())
								section_index = torch.where(section_index>0, copy.deepcopy(section_index), torch.zeros_like(section_index) + self.bins_attention + 1)
								#print(section_index.max(), section_index.min(), self.bins_attention + 3)
								#print ((section_index>self.bins_attention + 3).nonzero(as_tuple=True))
								temp_store_activations = (F.one_hot(section_index, num_classes = self.bins_attention + 3)).permute(0,5,1,2,3,4)
								
								# print(' temp storage activations: ', temp_storage_activations.size())
								if self.mask:
										# input mask is B X L X L
										# temp mask is of size bins X layers X heads X length X length
										#assert attention_mask.size() == (self.max_seq_len,self.max_seq_len)
										temp_mask = (attention_mask).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(temp_store_activations.device)
										#temp_mask = temp_mask.repeat(1,temp_store_activations.size(0), temp_store_activations.size(1), temp_store_activations.size(1), 1, 1)
										temp_store_activations = temp_store_activations*temp_mask
										del temp_mask
								temp_store_activations = torch.max (temp_store_activations, dim = 0).values
								self.coverage_attention_dicts += temp_store_activations 
								del temp_store_activations
								del current_coverage_tracker
				
				def _compute_intermediate_coverage(self):
								"""Calculate `neuron_coverage` for current model"""
								neuron_word_coverage, neuron_word_coverage_total = 0.0, 0.0
								neuron_attention_coverage, neuron_attention_coverage_total = 0.0, 0.0
								neuron_word_coverage += ( np.count_nonzero(self.coverage_word_dicts[:, 0:(self.bins_word+1), ...].numpy()) + np.count_nonzero(self.coverage_output_dicts[:, 0:(self.bins_word+1), ...].numpy()) )
								neuron_word_coverage_total += (self.coverage_word_dicts[:, 0:(self.bins_word+1), ...].numel() + self.coverage_output_dicts[:, 0:(self.bins_word+1), ...].numel())

								neuron_attention_coverage += np.count_nonzero(self.coverage_attention_dicts[:, 0:(self.bins_attention+1), ...].numpy())
								neuron_attention_coverage_total += self.coverage_attention_dicts[:, 0:(self.bins_attention+1), ...].numel()
								return neuron_word_coverage, neuron_word_coverage_total, neuron_attention_coverage, neuron_attention_coverage_total
				def _compute_coverage(self):			
								neuron_word_coverage, neuron_word_coverage_total, neuron_attention_coverage, neuron_attention_coverage_total = self._compute_intermediate_coverage()
								neuron_coverage = neuron_word_coverage + self.alpha*neuron_attention_coverage
								# print('Word and Attention Only: ', neuron_word_coverage , neuron_attention_coverage)
								neuron_coverage_total = neuron_word_coverage_total	+ self.alpha*neuron_attention_coverage_total 
								# print('Total Word and Attention Only: ', neuron_word_coverage_total , neuron_attention_coverage_total)
								
								return neuron_coverage/neuron_coverage_total

				def _compute_vector(self):
								"""Calculate `neuron_coverage` for current model"""
								neuron_coverage_vector = []
								for section in self.coverage_word_dicts:
												for entry in section.values():
														neuron_coverage_vector += ([entry_val.item() for entry_val in entry.flatten()])
								for section in self.coverage_attention_dicts:
												for entry in section.values():
														neuron_coverage_vector += ([entry_val.item() for entry_val in entry.flatten()])

								return neuron_coverage_vector
				def _update_coverage(self, text, word_mask = None):
								"""Update `coverage_tracker` for input `text`
												Args:
								`text`(str): text to update neuron coverage of.

								"""

								self._update_word_coverage(text, word_mask)
								
								self._update_attention_coverage(text, attention_mask)
								
				def __call__(self, testset, testset2=None, bz = 1):
								"""
								Returns neuron of `testset`
								Args:
												testset: Iterable of strings
								Returns:
												neuron coverage (float)
								"""
								# # # print('*'*50)
								
								# # # print('Updating Coverage using test set: ')
								mask_no, start = 0, 0
								with torch.no_grad():
										for t in tqdm(testset):
												
												
												if mask_no + bz >= len(testset):
														end = len(testset)
												else:
														end = start + bz
												if start >=	end or start >= len(testset) : break
												if testset2 is None:

																encodings = self.test_model.tokenizer(testset[start:end], padding='max_length', truncation=True, return_tensors="pt", max_length = self.max_seq_len)
						
												

																#print([v for k,v in encodings.data.items()])
																#encodings = {k: torch.cat([torch.LongTensor(c).unsqueeze(0).to(textattack.shared.utils.device) for c in v], dim = 0) for k, v in encodings.data.items()}
																if self.max_seq_len > 0:
																				input_ids = encodings.input_ids[:, : self.max_seq_len]
																				attention_mask = encodings.attention_mask[:, : self.max_seq_len]
																				token_types = False
																				if hasattr(encodings, 'token_type_ids'):
																								token_types = True
																								token_type_ids = encodings.token_type_ids[:, :self.max_seq_len]
																								token_type_ids = token_type_ids.to(textattack.shared.utils.device)
												else:
																
																				#input_text_list = [t[0] for t in trainset[start:end]]
																				print(testset[start:end],testset2[start:end])
																				encodings = self.test_model.tokenizer.batch_encode_plus(
																								testset[start:end],testset2[start:end],
																								truncation=True,
																								return_token_type_ids=True, 
																								max_length=self.max_seq_len,
																								padding="max_length",
																						)
												

																				#print([v for k,v in encodings.data.items()])
																				encodings = {k: torch.cat([torch.LongTensor(c).unsqueeze(0).to(textattack.shared.utils.device) for c in v], dim = 0) for k, v in encodings.data.items()}
																				if self.max_seq_len > 0:
																								input_ids = encodings['input_ids'][:, : self.max_seq_len]
																								attention_mask = encodings['attention_mask'][:, : self.max_seq_len]
																								token_types = False
																								if 'token_type_ids' in encodings.keys():
																												token_types = True
																												token_type_ids = encodings['token_type_ids'][:, :self.max_seq_len]
																												token_type_ids = token_type_ids.to(textattack.shared.utils.device)
												
												
												input_ids = input_ids.to(textattack.shared.utils.device)
												attention_mask = attention_mask.to(textattack.shared.utils.device)
												word_masks = []
												interaction_masks = []
												if self.mask:
														for i in range((input_ids.size(0))):
																encodings_temp = input_ids[i,...]
																current_mask = torch.zeros_like(encodings_temp)
																current_att_mask = torch.zeros((encodings_temp.size(0), encodings_temp.size(0)))
																
																for k,enc in enumerate(encodings_temp.tolist()):
																		
																				current_mask[k] = self.word_importance[enc]
																				current_att_mask[k,k] = 1
																				if enc == self.PAD: break
																				for j in range(k,len(encodings_temp.tolist())):
																				#print(encodings_temp, i,j,encodings_temp.tolist()[i],encodings_temp.tolist()[j])
																								current_att_mask[k,j] == self.interaction_importance[enc, encodings_temp[j]]
																								current_att_mask[j,k] == current_att_mask[k,j]
																								if encodings_temp[j] == self.PAD: break
																				word_masks.append(current_mask)
																				interaction_masks.append(current_att_mask)
																del current_att_mask, current_mask


												if token_types:
																outputs = self.test_model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,	output_attentions=True, output_hidden_states = True)
												else:
																outputs = self.test_model(input_ids, attention_mask=attention_mask,	output_attentions=True, output_hidden_states = True)
												all_hidden_states, all_attentions = outputs[-2:]
												self._update_output_coverage(outputs[0])
												if self.mask:
																self._update_word_coverage(all_hidden_states, torch.cat([wm.unsqueeze(0) for wm in word_masks], dim = 0))
																self._update_attention_coverage(all_attentions , torch.cat([wm.unsqueeze(0) for wm in interaction_masks], dim = 0))
												else:
																self._update_word_coverage(all_hidden_states)
																self._update_attention_coverage(all_attentions)
												


												del word_masks
												del interaction_masks
												
								
												start = end

								

												
								# # # print('*'*50)
								# # # print()
								# # # print('*'*50)
								# # # print('Computing Coverage: ')
								neuron_coverage = self._compute_coverage()
								# # # print('*'*50)
								return neuron_coverage
				def vector(self, testset, start = False):
								"""
								Returns neuron of `testset`
								Args:
												testset: Iterable of strings
								Returns:
												neuron coverage (float)
								"""
								# # # print('*'*50)
								if start:
										self.coverage_word_dicts = copy.deepcopy(self.training_word_coverage_dicts)
										self.coverage_attention_dicts = copy.deepcopy(self.training_attention_coverage_dicts)
								# # # print('Updating Coverage using test set: ')
								# # # print('#'*100)
								# # # print(len(testset))
								# # # print(testset)
								# # # print('#'*100)
								for t in tqdm(testset):
												# # # print(t)
												self._update_coverage(t)
								
								# # # print('*'*50)
								# # # print()
								# # # print('*'*50)
								# # # print('Computing Coverage: ')
								neuron_coverage = self._compute_vector()
								# # print('*'*50)
								return neuron_coverage
