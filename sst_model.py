import torch
import torch.nn as nn
import textattack
import transformers
class SSTModel(nn.Module):
	def __init__(self, model_name_or_path, max_seq_len):
		super().__init__()
		# the pretrained test model configuration 
		test_model = model_name_or_path
		self.config = transformers.AutoConfig.from_pretrained(
					test_model, output_hidden_states=True
			)
		#	the pretrained model , that needs to be tested 
		self.test_model = (
				transformers.AutoModelForSequenceClassification.from_pretrained(
						test_model, config=self.config
				)
		)
		# the tokenizer used by the test model 
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(
				test_model, use_fast=True
		)
		self.max_seq_len = max_seq_len
		vocab_size = self.tokenizer.vocab_size
		# bert/roberta etc	
		# the level of sparse to be used	
		self.test_model = self.test_model.to(textattack.shared.utils.device)

	def forward(self, text):
		# getting the token ids , input form used by BERT 
		encodings = self.tokenizer.batch_encode(text, return_tensors="pt", padding = True, truncation = True)
		encodings = encodings.to(textattack.shared.utils.device)
		if self.max_seq_len > 0:
				input_ids = encodings.input_ids[:, : self.max_seq_len]
				attention_mask = encodings.attention_mask[:, : self.max_seq_len]
				token_ids = encodings.token_type_ids[:, : self.max_seq_len]
		
		input_ids = input_ids.to(textattack.shared.utils.device)
		attention_mask = attention_mask.to(textattack.shared.utils.device)
		#token_type_ids = token_type_ids.to(textattack.shared.utils.device)
		# the predictions by the test model 
		model_outputs = self.test_model(input_ids, attention_mask=attention_mask, token_type_ids = token_ids)
		# B X L X E 

		model_outputs = torch.softmax(model_outputs[0], dim = -1)
		#print(model_outputs)
		# get model outputs 
		# 1 is pos, 0 is neg
		return model_outputs, torch.argmax(model_outputs, dim=-1)
