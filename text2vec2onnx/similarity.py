import numpy as np


def topk(array, k, axis=-1, sorted=True):
    # Use np.argpartition is faster than np.argsort, but do not return the values in order
    # We use array.take because you can specify the axis
    partitioned_ind = (
        np.argpartition(array, -k, axis=axis)
        .take(indices=range(-k, 0), axis=axis)
    )
    # We use the newly selected indices to find the score of the top-k values
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)

    if sorted:
        # Since our top-k indices are not correctly ordered, we can sort them with argsort
        # only if sorted=True (otherwise we keep it in an arbitrary order)
        sorted_trunc_ind = np.flip(
            np.argsort(partitioned_scores, axis=axis), axis=axis
        )

        # We again use np.take_along_axis as we have an array of indices that we use to
        # decide which values to select
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores

    return scores, ind


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)

    if not isinstance(b, np.ndarray):
        b = np.array(b)

    if len(a.shape) == 1:
        a = a.reshape(1, -1)

    if len(b.shape) == 1:
        b = b.reshape(1, -1)

    # Normalize the vectors
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

    # Compute cosine similarity
    return np.dot(a_norm, b_norm.T)


def semantic_search(query_embeddings,
                    corpus_embeddings,
                    query_chunk_size=100,
                    corpus_chunk_size=500000,
                    top_k=10,
                    score_function=cos_sim):
    """
   This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional nparray with the query embeddings.
    :param corpus_embeddings: A 2 dimensional nparray with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Funtion for computing scores. By default, cosine similarity.
    :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
    """
    if isinstance(query_embeddings, list):
        query_embeddings = np.array(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.reshape(1, -1)

    if isinstance(corpus_embeddings, list):
        corpus_embeddings = np.array(corpus_embeddings)

    queries_result_list = [[] for _ in range(len(query_embeddings))]
    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])
            cos_scores_top_k_values, cos_scores_top_k_idx = topk(cos_scores, min(top_k, len(cos_scores[0])), axis=1,
                                                                 sorted=False)
            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})
    # Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list
