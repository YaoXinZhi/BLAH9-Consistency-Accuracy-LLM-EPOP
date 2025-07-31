# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 19/03/2025 20:18
@Author: yao
"""

import os
import numpy as np

from icecream import ic

import argparse

from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.stats.multitest import multipletests
from collections import defaultdict


def read_doc_len_file(doc_length_file: str):
    doc_to_len = {}

    with open(doc_length_file) as f:
        for line in f:
            l = line.strip().split('\t')
            doc_idx = l[ 0 ]
            length = int(l[ 1 ])

            doc_to_len[ doc_idx ] = length

    return doc_to_len


def read_entity_relation_num_file(entity_relation_num_file: str):
    document_to_entity_num = {}
    document_to_relation_num = {}

    document_to_entity_dup_num = {}
    document_to_relation_dup_num = {}

    with open(entity_relation_num_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            document = l[ 0 ]
            entity_num = int(l[ 1 ])
            relation_num = int(l[ 2 ])

            entity_dup_num = int(l[ 3 ])
            relation_dup_num = int(l[ 4 ])

            document_to_entity_num[ document ] = entity_num
            document_to_relation_num[ document ] = relation_num

            document_to_entity_dup_num[ document ] = entity_dup_num
            document_to_relation_dup_num[ document ] = relation_dup_num

    return document_to_entity_num, document_to_relation_num, document_to_entity_dup_num, document_to_relation_dup_num


# 每一列是一个模型
# 每一行是一个文档
def read_eval_file(eval_file: str, sorted_document_list: list = None):
    model_to_score_list = defaultdict(list)
    document_list = [ ]
    with open(eval_file) as f:
        # fixme: 现在还没有gpt
        model_list = f.readline().strip().split('\t')
        for line in f:
            l = line.strip().split('\t')
            document = l[ 0 ]
            score_list = l[1:]

            # fixme: 如果有format_err 直接跳过
            if 'format_err' in score_list:
                continue

            score_list = list(map(float, score_list))

            document_list.append(document)
            for idx, model in enumerate(model_list):
                model_to_score_list[ model ].append(score_list[ idx ])

    new_model_to_score_list = {}
    if sorted_document_list:
        for model, score_list in model_to_score_list.items():
            # new_score_list = [ score_list[ document_list.index(i) ] for i in sorted_document_list ]
            new_document_list = [doc for doc in sorted_document_list if doc in document_list]
            new_score_list = [ score_list[ document_list.index(i) ] for i in sorted_document_list if i in document_list]
            new_model_to_score_list[ model ] = new_score_list
        # return new_model_to_score_list, model_list, sorted_document_list
        return new_model_to_score_list, model_list, new_document_list
    else:
        model_to_score_list = model_to_score_list.copy()

        return model_to_score_list, model_list, document_list


def multi_correlation_test(factor_list_1: list, factor_list_2: list,
                           model: str, factor_1: str, factor_2: str):
    factor_list_1 = np.nan_to_num(factor_list_1, nan=0)
    factor_list_2 = np.nan_to_num(factor_list_2, nan=0)

    # 计算 Pearson 相关系数
    pearson_corr, pearson_p = pearsonr(factor_list_1, factor_list_2)

    # 计算 Spearman 相关系数
    spearman_corr, spearman_p = spearmanr(factor_list_1, factor_list_2)

    # 计算 Kendall's Tau 相关系数
    kendall_corr, kendall_p = kendalltau(factor_list_1, factor_list_2)

    write_list = list(map(str, [ model, factor_1, factor_2,
                                 pearson_corr, pearson_p,
                                 spearman_corr, spearman_p,
                                 kendall_corr, kendall_p,
                                 '\n' ]))

    return write_list


def main():



    parser = argparse.ArgumentParser()
    parser.add_argument('--f1_file',
                        default='all-model.main-F1.train-dev.tsv',
                        help='default: all-model.main-F1.train-dev.tsv.')

    parser.add_argument('--kappa_file',
                        default='all-model.main-Kappa.tsv',
                        help='default: all-model.main-Kappa.tsv.')

    parser.add_argument('--save_file',
                        default='correlation-test-result.tsv',
                        help='default: correlation-test-result.tsv')

    parser.add_argument('--base_path',
                        default='/Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/output/evaluation/main-experiment_train_dev_v3',
                        help='default: /Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/output/evaluation/main-experiment_train_dev_v3')

    args = parser.parse_args()


    base_path = args.base_path

    save_file = f'{base_path}/{args.save_file}'

    f1_file = f'{base_path}/{args.f1_file}'
    kappa_file = f'{base_path}/{args.kappa_file}'

    entity_relation_num_file = f'{base_path}/entities_relation_statistics.EPOP.tsv'
    doc_length_file = f'{base_path}/doc_length.EPOP.tsv'

    print('start running.')
    document_to_entity_num, document_to_relation_num, document_to_entity_dup_num, document_to_relation_dup_num = read_entity_relation_num_file(
        entity_relation_num_file)
    sorted_document_list = list(sorted(document_to_entity_num.keys()))

    doc_to_len = read_doc_len_file(doc_length_file)

    # model_to_f1_list, _, _ = read_eval_file(f1_file, sorted_document_list)
    # model_to_kappa_list, _, _ = read_eval_file(kappa_file, sorted_document_list)
    # fixme: update document list
    model_to_f1_list, _, sorted_document_list = read_eval_file(f1_file, sorted_document_list)
    model_to_kappa_list, _, sorted_document_list = read_eval_file(kappa_file, sorted_document_list)

    model_list = [ 'gpt-4o-mini', 'kimi', 'deepseek', 'qwen3' ]

    doc_len_list = [ doc_to_len[ doc_idx ] for doc_idx in sorted_document_list ]
    entity_num_list = [ document_to_entity_num[ doc_idx ] for doc_idx in sorted_document_list ]
    relation_num_list = [ document_to_relation_num[ doc_idx ] for doc_idx in sorted_document_list ]
    entity_num_dup_list = [ document_to_entity_dup_num[ doc_idx ] for doc_idx in sorted_document_list ]
    relation_num_dup_list = [ document_to_relation_dup_num[ doc_idx ] for doc_idx in sorted_document_list ]

    with open(save_file, 'w') as wf:
        wf.write('model\tfactor-1\tfactor-2\t'
                 'pearson-corr\tpearson-p\t'
                 'spearman-corr\tspearman-p\t'
                 'kendall-corr\tkendall-p\n')
        for model in model_list:
            print(f'processing {model}.')

            f1_list = model_to_f1_list[ model ]
            kappa_list = model_to_kappa_list[ model ]

            # 准确性和一致性
            wf.write('\t'.join(multi_correlation_test(f1_list, kappa_list, model, 'Accuracy', 'Consistency')))

            # 复杂性和准确性
            wf.write('\t'.join(multi_correlation_test(f1_list, entity_num_list, model, 'Accuracy', '# of Entities')))
            wf.write('\t'.join(
                multi_correlation_test(f1_list, entity_num_dup_list, model, 'Accuracy', '# of Entities (duplicated)')))

            wf.write('\t'.join(multi_correlation_test(f1_list, relation_num_list, model, 'Accuracy', '# of Relation')))
            wf.write('\t'.join(multi_correlation_test(f1_list, relation_num_dup_list, model, 'Accuracy',
                                                      '# of Relation (duplicated)')))

            # 一致性和准确性
            wf.write(
                '\t'.join(multi_correlation_test(kappa_list, entity_num_list, model, 'Consistency', '# of Entities')))
            wf.write('\t'.join(multi_correlation_test(kappa_list, entity_num_dup_list, model, 'Consistency',
                                                      '# of Entities (duplicated)')))

            wf.write(
                '\t'.join(multi_correlation_test(kappa_list, relation_num_list, model, 'Consistency', '# of Relation')))
            wf.write('\t'.join(multi_correlation_test(kappa_list, relation_num_dup_list, model, 'Consistency',
                                                      '# of Relation (duplicated)')))

            # 准确性和文档长度
            wf.write('\t'.join(multi_correlation_test(f1_list, doc_len_list, model, 'Accuracy', 'Document Length')))
            # 一致性和文档长度
            wf.write(
                '\t'.join(multi_correlation_test(kappa_list, doc_len_list, model, 'Consistency', 'Document Length')))

        print('processing complexity & doc-length')
        # 复杂性和文档长度
        wf.write(
            '\t'.join(multi_correlation_test(doc_len_list, entity_num_list, '-', 'Document Length', '# of Entities')))
        wf.write('\t'.join(multi_correlation_test(doc_len_list, entity_num_dup_list, '-', 'Document Length',
                                                  '# of Entities (duplicated)')))

        wf.write(
            '\t'.join(multi_correlation_test(doc_len_list, relation_num_list, '-', 'Document Length', '# of Relation')))
        wf.write('\t'.join(multi_correlation_test(doc_len_list, relation_num_dup_list, '-', 'Document Length',
                                                  '# of Relation (duplicated)')))

        print('processing # of Entities & # of Relation')
        # 实体数量和关系数量
        wf.write('\t'.join(
            multi_correlation_test(relation_num_list, entity_num_list, '-', '# of Relation', '# of Entities')))
        wf.write('\t'.join(multi_correlation_test(relation_num_list, entity_num_dup_list, '-', '# of Relation',
                                                  '# of Entities (duplicated)')))
        wf.write('\t'.join(
            multi_correlation_test(relation_num_dup_list, entity_num_list, '-', '# of Relation (duplicated)',
                                   '# of Entities')))

    print(f'{save_file} saved.')


if __name__ == '__main__':
    main()
