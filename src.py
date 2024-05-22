import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

def store_embeddings(words_path: str, file_path: str, model: str = "text-embedding-3-large") -> None:
    """
    Store embeddings for a list of words using the OpenAI API and save the results to a CSV file.

    Args:
        words_path (str): The path to the file containing the list of words.
        file_path (str): The path to the CSV file where the embeddings will be saved.
        model (str, optional): The name of the model to use for generating embeddings. Defaults to "text-embedding-3-large".

    Returns:
        None
    """
    with open("api_key.txt", encoding="utf-8") as f:
        key = f.read()
    client = OpenAI(api_key=key)
    # Initialize an empty DataFrame with appropriate column names
    df = pd.DataFrame(columns=['word', 'embedding'])

    # Read the list of words from the file
    df = pd.read_csv(words_path)
    words = df['word'].tolist()

    # Process each word and store the results in the DataFrame
    for i in tqdm(range(len(words))):
        if not isinstance(words[i], str):
            print(words[i])
            continue
        embedding = client.embeddings.create(input=words[i], model=model).data[0].embedding
        df = df.append({'word': words[i], 'embedding': embedding}, ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False, encoding='utf-8')

def load_dict_from_csv(file_path : str) -> dict[str, list[float]]:
    """
    Load a dictionary from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary created from the CSV file, where the first column is used as keys and the second column is used as values.
    """
    df = pd.read_csv(file_path, header=None)
    loaded_dict = pd.Series(df[1].values, index=df[0]).to_dict()
    return loaded_dict


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    v1 (list[float]): The first vector.
    v2 (list[float]): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    

def neighbors(word: str, dict: dict[str:list[float]], model: str = "text-embedding-3-large", n_neighbors: int = 100, api_key: bool = True) -> dict[str, float]:
    """
    Find nearest or farthest neighbors of a word based on its embedding similarity.

    Args:
        word (str): The word for which neighbors need to be found.
        dict (dict[str:list[float]]): A dictionary containing word embeddings.
        model (str, optional): The name of the embedding model to use. Defaults to "text-embedding-3-large".
        n_neighbors (int, optional): The number of neighbors to find. Defaults to 100.
        api_key (bool, optional): Whether to use an API key for computing embeddings. Defaults to True.
        nearest (bool, optional): Whether to find nearest neighbors. If False, farthest neighbors will be found. Defaults to True.

    Returns:
        dict[str, float]: A dictionary containing the word and its similarity score with the neighbors.
    """
    
    # Check if the word is in the dictionary and compute its embedding if not
    if word not in dict:
        print(f"Word '{word}' not found in dictionary")
        if api_key:
            print("Computing its embedding")
            with open("api_key.txt", encoding="utf-8") as f:
                key = f.read()
            client = OpenAI(api_key=key)
            word_embedding = client.embeddings.create(input=word, model=model).data[0].embedding
        else: 
            print("Word not found in dictionary and no API key provided")
            return
    else:
        word_embedding = dict[word]
       
    # Calculate the similarity between the word and all other words in the dictionary
    similarity = {}
    for key in dict:
        similarity[key] = cosine_similarity(word_embedding, dict[key])

    # Sort the results by similarity and return the top n_neighbors
    result = sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:n_neighbors]
    df = pd.DataFrame(result, columns=["word", "similarity"])
    if word[0].isupper():
        word = word[0].lower() + word[1:] + "_maj"
    df.to_csv(f"nearest_neighbors/{word}.csv", index=False)

    return result