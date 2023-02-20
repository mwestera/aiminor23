import transformers

"""
Script illustrating text generation using a pre-trained transformer (GPT2) from the Huggingface ecosystem.
Closely based on:  https://huggingface.co/blog/how-to-generate
"""

def main():

    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    prompt = 'I enjoy walking with my cute dog'
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output_ids = model.generate(
        input_ids,
        max_length=60,
        # num_beams=5,
        # early_stopping=True,
        # no_repeat_ngram_size=2,
        # num_return_sequences=5,

        # https://blog.fastforwardlabs.com/images/2019/05/Screen_Shot_2019_05_08_at_3_06_36_PM-1557342561886.png
        # Ari Holtzman et al. (2019)

        # output_scores=True, return_dict_in_generate=True,

        # do_sample=True,
        # temperature=0.7,
        # top_k=50,
        # top_p=0.92,
    )

    for n, output in enumerate(output_ids):
        print(f"\n= = = = = = Output {n} = = = = = = =")
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(text)




if __name__ == '__main__':
    main()