import argparse
from logging import getLogger
from recbole.config import Config
from recbole.utils import set_color
from recbole.utils import init_seed, init_logger
from model.vqrec import VQRec
# from model.vqrec_tid import VQRec
from model.hgn import HGN
from data.dataset import PretrainVQRecDataset
import torch
from recbole.data import data_preparation
from trainer import VQRecTrainer


def finetune(model_name, dataset, pretrained_file='', finetune_mode='', **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=VQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    dataset = PretrainVQRecDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = VQRec(config, train_data.dataset).to(config['device'])
    model.pq_codes = model.pq_codes.to(config['device'])
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if finetune_mode == 'fix_enc':
            logger.info('[Fine-tune mode] Fix Seq Encoder!')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    logger.info(model)
    trainer = VQRecTrainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
    # best_valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=True)

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')

    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='VQRec', help='model name')
    parser.add_argument('-d', type=str, default='P', help='dataset name')
    parser.add_argument('-p', type=str, default='save_OP/save_OP/Pantry_single.pth', help='pre-trained model path')
    parser.add_argument('-f', type=str, default='', help='fine-tune mode')
    args, unparsed = parser.parse_known_args()
    print(args)
    finetune(args.m, args.d, pretrained_file=args.p, finetune_mode=args.f)
# saved_single/VQRec-Jun-18-2023_09-28-27.pth
# saved_single/VQRec-Jun-21-2023_14-15-40.pth
# save_CI/save_CI_sparse/VQRec-Jul-21-2023_16-29-03.pth