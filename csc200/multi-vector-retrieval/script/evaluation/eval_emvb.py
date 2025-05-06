import numpy as np
import os
from typing import Dict
import sys
import time
from typing import Dict, Callable
import json
import copy

import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from baseline.emvb.script import build_index, util, performance_metric, evaluation_pipeline


def approximate_solution_retrieval(username: str, dataset: str, topk: int, build_index_suffix: str,
                                   retrieval_config: dict):
    nprobe = retrieval_config['nprobe']
    thresh = retrieval_config['thresh']
    out_second_stage = retrieval_config['out_second_stage']
    thresh_query = retrieval_config['thresh_query']
    n_doc_to_score = retrieval_config['n_doc_to_score']
    print(f"retrieval: nprobe {nprobe}, thresh {thresh}, out_second_stage {out_second_stage}, "
          f"thresh_query {thresh_query}, n_doc_to_score {n_doc_to_score}")

    retrieval_suffix = f'nprobe_{nprobe}-thresh_{thresh}-out_second_stage_{out_second_stage}-thresh_query_{thresh_query}-n_doc_to_score_{n_doc_to_score}'

    os.system(
        f'cd /home/{username}/multi-vector-retrieval/baseline/emvb/build && ./perf_emvb -topk {topk} -nprobe {nprobe} -thresh {thresh} '
        f'-out-second-stage {out_second_stage} -thresh-query {thresh_query} -n-doc-to-score {n_doc_to_score} '
        f'-username {username} -dataset {dataset} -build-index-suffix {build_index_suffix} -retrieval-suffix {retrieval_suffix}')

    result_fname = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer/' \
                   f'{dataset}-emvb-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'
    performance_fname = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer/' \
                        f'{dataset}-emvb-performance-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'

    performance_df = pd.read_csv(performance_fname, delimiter='\t')

    search_time_m = {
        "retrieval_time_single_query_p5(ms)": '{:.3f}'.format(np.percentile(performance_df['search_time'], 5)),
        "retrieval_time_single_query_p50(ms)": '{:.3f}'.format(np.percentile(performance_df['search_time'], 50)),
        "retrieval_time_single_query_p95(ms)": '{:.3f}'.format(np.percentile(performance_df['search_time'], 95)),
        "retrieval_time_single_query_average(ms)": '{:.3f}'.format(1.0 * np.average(performance_df['search_time'])),

        "cand_doc_retrieval_time_average(ms)": '{:.3f}'.format(
            1.0 * np.average(performance_df['cand_doc_retrieval_time'])),
        "doc_filtering_time_average(ms)": '{:.3f}'.format(1.0 * np.average(performance_df['doc_filtering_time'])),
        "second_stage_time_average(ms)": '{:.3f}'.format(1.0 * np.average(performance_df['second_stage_time'])),
        "doc_scoring_time_average(ms)": '{:.3f}'.format(1.0 * np.average(performance_df['doc_scoring_time'])),

        "n_cand_doc_retrieval_average": '{:.3f}'.format(1.0 * np.average(performance_df['n_cand_doc_retrieval'])),
        "n_doc_filtering_average": '{:.3f}'.format(1.0 * np.average(performance_df['n_doc_filtering'])),
        "n_second_stage_average": '{:.3f}'.format(1.0 * np.average(performance_df['n_second_stage'])),
        "n_vq_score_refine_average": '{:.3f}'.format(1.0 * np.average(performance_df['n_vq_score_refine'])),
    }

    retrieval_time_ms_l = performance_df['search_time']

    return retrieval_suffix, search_time_m, retrieval_time_ms_l


def approximate_solution_build_index(username: str, dataset: str,
                                     build_index_config: dict, build_index_suffix: str):
    print(f"start build index")

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
    n_vecs = np.sum(item_n_vec_l)
    n_centroid = build_index_config['n_centroid_f'](n_vecs)
    pq_n_partition = build_index_config['pq_n_partition']
    pq_n_bit_per_partition = build_index_config['pq_n_bit_per_partition']

    start_time = time.time()
    _, record_build_index = build_index.build_index(username=username, dataset=dataset, n_centroid=n_centroid,
                                                         pq_n_partition=pq_n_partition,
                                                         pq_n_bit_per_partition=pq_n_bit_per_partition)

    end_time = time.time()
    build_index_time_sec = end_time - start_time
    print(f"insert time spend {build_index_time_sec:.3f}s")
    print(f"n_centroid {n_centroid}, total_n_vec {n_vecs}")

    module_name = 'emvb'

    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
    build_index_performance_filename = os.path.join(result_performance_path,
                                                    f'{dataset}-build_index-{module_name}-{build_index_suffix}.json')
    if record_build_index:
        with open(build_index_performance_filename, 'w') as f:
            json.dump({'build_index_time (s)': build_index_time_sec}, f)


def approximate_solution_retrieval_outter(username: str, dataset: str,
                                          build_index_config: Dict,
                                          build_index_suffix: str,
                                          topk: int,
                                          retrieval_parameter_l: list,
                                          retrieval_f: Callable,
                                          method_name: str = 'emvb',
                                          ):
    query_l, queryID_l = evaluation_pipeline.load_query(username=username, dataset=dataset)

    mrr_gnd, success_gnd, recall_gnd_id_m = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                                topk=topk)

    final_result_l = []
    for retrieval_config in retrieval_parameter_l:
        retrieval_suffix, search_time_m, time_ms_l = retrieval_f(
            username=username, dataset=dataset, topk=topk,
            build_index_suffix=build_index_suffix,
            retrieval_config=retrieval_config)

        result_answer_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer/'
        method_ans_name = f'{dataset}-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'
        retrieval_result_filename = os.path.join(result_answer_path, method_ans_name)

        performance_metric.read_method_tsv(base_dir=result_answer_path, dataset=dataset, method_name=method_name,
                                           topk=topk, build_index_suffix=build_index_suffix,
                                           retrieval_suffix=retrieval_suffix)

        recall_l, mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
            username=username, dataset=dataset, topk=topk,
            method_name=method_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
            mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)

        build_index_config = copy.deepcopy(build_index_config)
        if 'item_n_vec_l' in build_index_config:
            del build_index_config['item_n_vec_l']
        if 'item_n_vecs_l' in build_index_config:
            del build_index_config['item_n_vecs_l']
        retrieval_info_m = {'n_query': len(queryID_l), 'topk': topk,
                            'build_index': build_index_config, 'retrieval': retrieval_config,
                            'search_time': search_time_m, 'search_accuracy': search_accuracy_m}
        if 'n_centroid_f' in retrieval_info_m['build_index']:
            del retrieval_info_m['build_index']['n_centroid_f']
        method_performance_name = f'{dataset}-retrieval-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
        result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
        performance_filename = os.path.join(result_performance_path, method_performance_name)
        with open(performance_filename, "w") as f:
            json.dump(retrieval_info_m, f)

        df = pd.DataFrame({'time(ms)': time_ms_l, 'recall': recall_l})
        df.index.name = 'local_queryID'
        if mrr_l:
            df['mrr'] = mrr_l
        if success_l:
            df['success'] = success_l
        single_query_performance_name = f'{dataset}-retrieval-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.csv'
        result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/single_query_performance'
        single_query_performance_filename = os.path.join(result_performance_path, single_query_performance_name)
        df.to_csv(single_query_performance_filename, index=True)

        print("#############final result###############")
        print("filename", method_performance_name)
        print("search time", retrieval_info_m['search_time'])
        print("search accuracy", retrieval_info_m['search_accuracy'])
        print("########################################")

        final_result_l.append({'filename': method_performance_name, 'search_time': retrieval_info_m['search_time'],
                               'search_accuracy': retrieval_info_m['search_accuracy']})

    for final_result in final_result_l:
        print("#############final result###############")
        print("filename", final_result['filename'])
        print("search time", final_result['search_time'])
        print("search accuracy", final_result['search_accuracy'])
        print("########################################")


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for nprobe in grid_search_para['nprobe']:
        for thresh in grid_search_para['thresh']:
            for out_second_stage in grid_search_para['out_second_stage']:
                for thresh_query in grid_search_para['thresh_query']:
                    for n_doc_to_score in grid_search_para['n_doc_to_score']:
                        parameter_l.append(
                            {"nprobe": nprobe, "thresh": thresh, 'out_second_stage': out_second_stage,
                             'thresh_query': thresh_query, 'n_doc_to_score': n_doc_to_score})
    return parameter_l


if __name__ == '__main__':
    # default value {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x))))},
    config_l = {
        'dbg': {
            'username': 'username2',
            'dataset_l': ['msmacro', 'lotte'],
            'topk_l': [10],
            'is_debug': False,
            'build_index_parameter_l': [
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))),
                 'pq_n_partition': 16, 'pq_n_bit_per_partition': 8}
            ],
            'retrieval_parameter_l': [
                {'nprobe': 4, 'thresh': 0.4, 'out_second_stage': 512, 'thresh_query': 0.5, 'n_doc_to_score': 4000},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'nprobe': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    'thresh': [0.3],
                    'out_second_stage': [256],
                    'thresh_query': [0.3],
                    'n_doc_to_score': [500]

                    # 'nprobe': [1, 6],
                    # 'thresh': [0.3, 0.7],
                    # 'out_second_stage': [512, 2048],
                    # 'thresh_query': [0.3, 0.7],
                    # 'n_doc_to_score': [20, 100, 1000, 4000]
                },
                100: {
                    'nprobe': [2],
                    'thresh': [0.3, 0.7],
                    'out_second_stage': [512, 2048],
                    'thresh_query': [0.3, 0.7],
                    'n_doc_to_score': [100, 500, 4000]
                },
                1000: {
                    'nprobe': [4, 5, 6, 7],
                    'thresh': [0.4],
                    'out_second_stage': [512],
                    'thresh_query': [0.5],
                    'n_doc_to_score': [4000]
                }
            }
        },
        'local': {
            'username': 'username1',
            # 'dataset_l': ['fake-normal', 'lotte-500-gnd'],
            'dataset_l': ['lotte-500-gnd'],
            # 'dataset_l': ['lotte-lifestyle'],
            # 'topk_l': [10, 50],
            'topk_l': [10],
            'is_debug': True,
            'build_index_parameter_l': [
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))),
                 'pq_n_partition': 32, 'pq_n_bit_per_partition': 8}
            ],
            'retrieval_parameter_l': [
                {'nprobe': 4, 'thresh': 0.4, 'out_second_stage': 50, 'thresh_query': 0.5, 'n_doc_to_score': 100},
            ],
            'grid_search': False,
            'grid_search_para': {
                10: {
                    'nprobe': [4, 6],
                    'thresh': [0.4],
                    'out_second_stage': [50],
                    'thresh_query': [0.5],
                    'n_doc_to_score': [100]
                },
                50: {
                    'nprobe': [4, 6],
                    'thresh': [0.4],
                    'out_second_stage': [50],
                    'thresh_query': [0.5],
                    'n_doc_to_score': [100]
                },
            }
        }
    }
    host_name = 'local'
    config = config_l[host_name]

    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']
    is_debug = config['is_debug']
    build_index_parameter_l = config['build_index_parameter_l']

    util.compile_file(username=username, is_debug=is_debug)
    for dataset in dataset_l:
        for build_index_config in build_index_parameter_l:
            embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
            vec_dim = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy')).shape[1]
            n_item = np.load(os.path.join(embedding_dir, f'doclens.npy')).shape[0]
            item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
            n_vecs = np.sum(item_n_vec_l)

            n_centroid = build_index_config['n_centroid_f'](n_vecs)
            pq_n_partition = build_index_config['pq_n_partition']
            pq_n_bit_per_partition = build_index_config['pq_n_bit_per_partition']

            build_index_suffix = f'n_centroid_{n_centroid}-pq_n_partition_{pq_n_partition}'

            approximate_solution_build_index(
                username=username, dataset=dataset,
                build_index_config=build_index_config, build_index_suffix=build_index_suffix)

            for topk in topk_l:
                grid_search = config['grid_search']
                if grid_search:
                    retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'][topk])
                else:
                    retrieval_parameter_l = config['retrieval_parameter_l']

                approximate_solution_retrieval_outter(
                    username=username, dataset=dataset,
                    build_index_config=build_index_config,
                    build_index_suffix=build_index_suffix,
                    topk=topk,
                    retrieval_parameter_l=retrieval_parameter_l,
                    retrieval_f=approximate_solution_retrieval,
                    method_name='emvb'
                )
