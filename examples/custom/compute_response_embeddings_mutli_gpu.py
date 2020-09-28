"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

import argparse

from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import torch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_arg('-i', '--input-files', nargs='*', help='raw text corpus files containing `documents` to be embedded and converted to corpus of tensors')
    ap.add_arg('-t', '--tensor-file', help='file for saving corpus tensor, e.g. corpus_embeddings.pt')
    ap.add_arg('-s', '--sents-file', help='file for saving corpus sentences, e.g. corpus_sents.pkl')
    return ap.parse_args()

def get_docs(file):
    """
    get lines (docs) from input file
    """
    docs = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            docs.append(line.strip())
    return docs

#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    args = set_args()
    #Create a large list of 100k sentences
    # sentences = ["This is sentence {}".format(i) for i in range(100000)]
    sentences = []
    for file in args.input_files:
        sentences += get_docs(file)

    print(f'{len(sentences)} sentences collected from input files...')

    with open(args.sents_file, "wb") as f:
        pickle.dump(sentences, f)

    print(f'Sentences saved to {args.sents_file}')
    
    #Define the model
    model = SentenceTransformer('distiluse-base-multilingual-cased')

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    emb = model.encode_multi_process(sentences, pool)
    print("Embeddings computed. Shape:", emb.shape)

    torch.save(corpus_embeddings_ml, args.tensor_file)
    print(f'Embeddings saved to {args.tensor_file}')

    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
