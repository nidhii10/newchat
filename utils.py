from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
     

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# len(ids[0]) added to code so that question isn't included in chatbot response.

def generate_text(model_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length + len(ids[0]),
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(final_outputs[0, len(ids[0]):], skip_special_tokens=True)
    
    # Remove unwanted characters
    generated_text = generated_text.replace('\n', '').replace('\r', '').replace("\'s", "").replace('  ', '')
    
    return generated_text.strip()



model1_path = "C:/Users/Nidhi/Downloads/newchat/chatbot_project/model"

sequence1 = "Albert Einstein"
max_len = 50
x=generate_text(model1_path, sequence1, max_len)
print("\n" in x)
print(x)
print(x.replace('\n', '').replace('\r', '').replace("\'s", "").replace('  ', ''))
for i in x:
    print(i)