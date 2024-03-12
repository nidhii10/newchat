import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader
import os
import docx
     
 
# Functions to read different file types
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text
 
def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text
 
def read_txt(file_path):
    with open(file_path, "rb") as file:
        text = str(file.read())
    return text
 
def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif filename.endswith(".docx"):
            combined_text += read_word(file_path)
        elif filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text
 
     
 
# Read documents from the directory
train_directory = 'C:/Users/Nidhi/Downloads/newchat/chatbot_project/dataset'
#train_directory = '/content/drive/MyDrive/ColabNotebooks/data/chatbot_docs/training_data/q_and_a'
text_data = read_documents_from_directory(train_directory)
text_data = re.sub(r'\n+', '\n', text_data).strip()  # Remove excess newline characters
 
# Save the training and validation data as text files
with open("C:/Users/Nidhi/Downloads/newchat/chatbot_project/train_txt/train.txt", "w") as f:
   f.write(text_data)
     
 
#with open("/content/drive/MyDrive/ColabNotebooks/data/chatbot_docs/combined_text/q_and_a/train.txt", "w") as f:
#    f.write(text_data)
     
 
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
     
 
def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset
     
 
def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator
     
 
def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
  tokenizer = GPT2Tokenizer.from_pretrained(model_name)
  train_dataset = load_dataset(train_file_path, tokenizer)
  data_collator = load_data_collator(tokenizer)
 
  tokenizer.save_pretrained(output_dir)
     
  model = GPT2LMHeadModel.from_pretrained(model_name)
 
  model.save_pretrained(output_dir)
 
  training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
      )
 
  trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
  )
     
  trainer.train()
  trainer.save_model()
     
 
 
train_file_path = "C:/Users/Nidhi/Downloads/newchat/chatbot_project/train_txt/train.txt"
#train_file_path = "/content/drive/MyDrive/ColabNotebooks/data/chatbot_docs/combined_text/q_and_a/train.txt"
model_name = 'gpt2'
#output_dir = '/content/drive/MyDrive/ColabNotebooks/models/chat_models/custom_full_text'
output_dir = 'C:/Users/Nidhi/Downloads/newchat/chatbot_project/model'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 50.0
save_steps = 50000
     
 
# Train
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)
 
 
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
     
 
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model
 
 
def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer
 
def generate_text(model_path, sequence, max_length):
   
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
 
 
 
model1_path = "C:/Users/Nidhi/Downloads/newchat/chatbot_project/chatbot_project/model"
sequence1 = "What is the physics?"
max_len = 50
generate_text(model1_path, sequence1, max_len)
 
'''model2_path = "/content/drive/MyDrive/ColabNotebooks/models/chat_models/custom_q_and_a"
sequence2 = "[Q] What is the Babel fish?"
max_len = 50
generate_text(model2_path, sequence2, max_len) '''