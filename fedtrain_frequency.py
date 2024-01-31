
import argparse
import numpy as np
import torch.nn as nn
from logging import getLogger
from recbole.config import Config
from recbole.data.dataloader import TrainDataLoader
from recbole.utils import init_seed, init_logger
from FLtrainer.fedtrainer_ldp import FedtrainTrainer
from model.vqrec import VQRec
# from model.vqrec_tid import VQRec
from data.dataset import FederatedDataset
import os
import faiss
# import numpy as np
import torch

from utils import parse_faiss_index

def load_index(config, logger, item_num, field2id_token):
    code_dim = config['code_dim']
    code_cap = config['code_cap']
    dataset_name = config['dataset']
    index_suffix = config['index_suffix']
    if config['index_pretrain_dataset'] is not None:
        index_dataset = config['index_pretrain_dataset']
    else:
        index_dataset = config['dataset']
    index_path = os.path.join(
        config['index_path'],
        index_dataset,
        f'{index_dataset}.{index_suffix}'
    )
    logger.info(f'Index path: {index_path}')
    uni_index = faiss.read_index(index_path)
    pq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
    assert code_dim == pq_codes.shape[1], pq_codes.shape
    # assert item_num == 1 + pq_codes.shape[0], f'{item_num}, {pq_codes.shape}'
    # uint8 -> int32 to reserve 0 padding
    pq_codes = pq_codes.astype(np.int32)
    # 0 for padding
    pq_codes = pq_codes + 1
    # flatten pq codes
    base_id = 0
    for i in range(code_dim):
        pq_codes[:, i] += base_id
        base_id += code_cap + 1

    logger.info('Loading filtered index mapping.')
    filter_id_dct = {}
    with open(
            os.path.join(config['data_path'],
                         f'{dataset_name}.{config["filter_id_suffix"]}'),
            'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            filter_id_name = line.strip()
            filter_id_dct[filter_id_name] = idx

    logger.info('Converting indexes.')
    mapped_codes = np.zeros((item_num, code_dim), dtype=np.int32)
    for i, token in enumerate(field2id_token):
        if token == '[PAD]': continue
        mapped_codes[i] = pq_codes[filter_id_dct[token]]
    return torch.LongTensor(mapped_codes)

def count_frequency(tensor):
    unique_values, counts = torch.unique(tensor.view(-1), return_counts=True)
    frequency_dict = {}
    for value, count in zip(unique_values, counts):
        frequency_dict[value.item()] = count.item()
    return frequency_dict

def compute_weights(dict1, dict2, scope):
    max_value = max(max(dict1.values()), max(dict2.values()))
    weights_A = {}
    weights_B = {}

    for key in range(scope):
        value1 = dict1.get(key, 0)
        value2 = dict2.get(key, 0)

        if value1 == value2:
            weights_A[key] = 0.5
            weights_B[key] = 0.5
        else:
            total = value1 + value2
            weights_A[key] = value1 / total
            weights_B[key] = value2 / total

    return weights_A, weights_B

def pretrain(dataset, **kwargs):
    # configurations initialization
    props = ['props/VQRec.yaml', 'props/pretrain.yaml']
    print(props)

    # configurations initialization
    config = Config(model=VQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    config_A = Config(model=VQRec, dataset='O', config_file_list=props, config_dict=kwargs)
    config_B = Config(model=VQRec, dataset='A', config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    init_seed(config_A['seed'], config['reproducibility'])
    init_seed(config_B['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    init_logger(config_A)
    init_logger(config_B)
    logger = getLogger()
    logger.info(config)
    logger.info(config_A)
    logger.info(config_B)
    logger.info(dataset)

    dataset_A = FederatedDataset(config_A, pq_codes=None)
    logger.info(dataset_A)
    pretrain_dataset_A = dataset_A.build()[0]
    spilt_point = list(dataset_A.field2token_id['item_id'].items())[-1][1]
    dataset_B = FederatedDataset(config_B, pq_codes=None)

    logger.info(dataset_B)
    pretrain_dataset_B = dataset_B.build()[0]
    item_num = dataset_A.item_num + dataset_B.item_num - 1
    field2id_token = np.concatenate((dataset_A.field2id_token['item_id'], dataset_B.field2id_token['item_id'][1:]))
    pq_codes = load_index(config, logger, item_num, field2id_token).to(config['device'])
    item_pq_A = pq_codes[:spilt_point + 1]
    item_pq_B = torch.cat([pq_codes[0].unsqueeze(0), pq_codes[spilt_point + 1:]], dim=0)
    pretrain_dataset_A.pq_codes = item_pq_A
    pretrain_dataset_B.pq_codes = item_pq_B
    pretrain_data_A = TrainDataLoader(config_A, pretrain_dataset_A, None, shuffle=True)
    pretrain_data_B = TrainDataLoader(config_A, pretrain_dataset_B, None, shuffle=True)

    scope = config_A['code_dim'] * (1 + config_A['code_cap'])
    frequency_dict_A = count_frequency(item_pq_A)
    frequency_dict_B = count_frequency(item_pq_B)
    result_dict_A = {i: 0 for i in range(scope)}
    result_dict_B = {i: 0 for i in range(scope)}
    for key, value in frequency_dict_A.items():
        result_dict_A[key] += value
    for key, value in frequency_dict_B.items():
        result_dict_B[key] += value
    weights_A, weights_B = compute_weights(result_dict_A, result_dict_B, scope)
    weights_tensor_A = torch.tensor([weights_A[i] for i in range(scope)]).unsqueeze(1).repeat(1, config_A['hidden_size'])
    weights_tensor_B = torch.tensor([weights_B[i] for i in range(scope)]).unsqueeze(1).repeat(1, config_A['hidden_size'])
    clients_weights = torch.stack([weights_tensor_A.flatten(), weights_tensor_B.flatten()]).to(config['device'])

    model_A = VQRec(config_A, pretrain_data_A.dataset).to(config['device'])
    model_A.pq_codes.to(config['device'])
    logger.info(model_A)
    model_B = VQRec(config_B, pretrain_data_B.dataset).to(config['device'])
    model_B.pq_codes.to(config['device'])
    logger.info(model_B)
    global_embedding = nn.Embedding(
        config['code_dim'] * (1 + config['code_cap']), config['hidden_size'], padding_idx=0).to(config['device'])
    global_embedding.weight.data.normal_(mean=0.0, std=config['initializer_range'])
    model_A.pq_code_embedding.load_state_dict(global_embedding.state_dict())
    model_B.pq_code_embedding.load_state_dict(global_embedding.state_dict())
    trainer = FedtrainTrainer(config_A, config_B, model_A, model_B, global_embedding)
    trainer.fedtrain(pretrain_data_A, pretrain_data_B, clients_weights, show_progress=True)

    return config['model'], config['dataset']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='OA', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    model, dataset = pretrain(args.d)
