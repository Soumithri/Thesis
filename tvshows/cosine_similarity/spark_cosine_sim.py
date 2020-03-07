import os

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from tvshows.common.common import COSINE_SIM_DIR
from tvshows.common.common import USER_TOPIC_FILE

spark = SparkSession.builder \
    .master("local") \
    .appName("cos_sim") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

EXTERNAL_DIR = '/Volumes/Seagate Backup /pickle_dumps/'


def ensure_directory(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)


def find_matches_in_submatrix(sources, targets):
    cosimilarities = cosine_similarity(sources, targets)
    yield cosimilarities


def broadcast_matrix(mat):
    bcast = spark.sparkContext.broadcast((mat.data, mat.indices, mat.indptr))
    (data, indices, indptr) = bcast.value
    bcast_mat = csr_matrix((data, indices, indptr), shape=mat.shape)
    return bcast_mat


def parallelize_matrix(scipy_mat, rows_per_chunk=1000):
    [rows, cols] = scipy_mat.shape
    i = 0
    submatrices = []
    while i < rows:
        current_chunk_size = min(rows_per_chunk, rows - i)
        submat = scipy_mat[i:i + current_chunk_size]
        submatrices.append((i, (submat.data, submat.indices,
                                submat.indptr),
                            (current_chunk_size, cols)))
        i += current_chunk_size
    return spark.sparkContext.parallelize(submatrices)


def print_record(record):
    global start
    global end
    end += record.shape[0]
    df = pd.DataFrame(data=record, index=pandas_df.iloc[start:end, :].index,
                      columns=pandas_df.index[: NUM_ROWS]).astype(np.float32)
    df.to_pickle(EXTERNAL_DIR + 'dump_rows_{}_{}.pkl'.format(start, end))
    start = end
    print('Finished dumping cosine similarity for {}/{} users...'.format(start, len(pandas_df)))


pandas_df = pd.read_csv(USER_TOPIC_FILE, index_col='userid')
NUM_ROWS = len(pandas_df)

a_mat = csr_matrix(pandas_df.iloc[:NUM_ROWS, :].values)
b_mat = csr_matrix(pandas_df.iloc[:NUM_ROWS, :].values)

a_mat_para = parallelize_matrix(a_mat, rows_per_chunk=1000)
b_mat_dist = broadcast_matrix(b_mat)

cosine_sim_rdd = a_mat_para.flatMap(
    lambda submatrix: find_matches_in_submatrix(csr_matrix(submatrix[1],
                                                           shape=submatrix[2]),
                                                b_mat_dist,
                                                ))

start = 0
end = 0
pandas_df.columns = pandas_df.columns.astype(str)
pandas_df.index = pandas_df.index.astype(str)

ensure_directory(EXTERNAL_DIR)
cosine_sim_rdd.foreach(print_record)

spark.stop()
