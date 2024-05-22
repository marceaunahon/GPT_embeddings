# GPT_embeddings

**Paper:** to be published soon.

## Description 
This repository gathers the code used in the **Nearest neighbors** section of the paper.  
It allows to compute cosine similarity between words using the GPT-4 embedding model `text-embedding-3-large`, and run a nearest neighbors computation.
We computed the embedding of each word of the english dictionary. However, as the user may check using the [GPT4 tokenizer](https://platform.openai.com/tokenizer), the tokenization of a word depends on the context. For exemple, 'dignity' and ' dignity' do not have the same tokenization and thus they have different embeddings.  
In our dictionary, we gathered only four forms for each word: 
- the word itself ('dignity')
- the word preceded by a space (' dignity')
- the word with the first letter in capital ('Dignity') 
- the word preceded by a space with the first letter in capital (' Dignity')

There are many others forms ('DIGNITY', ':dignity', etc.), for which the provided code allows the user compute embeddings.

## Our results

The results of our paper (to be released soon) are displayed in the `nearest_neighbors` folder.

## Embedding dictionary

Due to storage limitations, we do not directly give access to our embedding dictionary (28Go). However, we can share it if you contact us.

## Install

**General Requirements:**
```bash
>> python -m pip install -r requirements.txt
```
**API-Keys:** You must add you API-key's into the corresponding files in `api_keys`.
