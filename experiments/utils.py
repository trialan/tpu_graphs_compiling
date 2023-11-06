import pickle

def save_as_pickle(graph_list, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(graph_list, f)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
