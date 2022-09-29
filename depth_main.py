from typing import List
import numpy as np
import torch as torch
from RCOL_datasets.datasets import DataGeneratorFactory, DataSet, LazyDepthDataSet, DepthDataGenerator
from helper import set_seeds
from hyperparameters import get_best_hyperparams_from_args, Hyperparameters
import argparse
import os
import warnings
import ast
import matplotlib
from sys import platform
from depth_results_helper import args_to_txt, save_performance_metrics
from utils.Calibration.I2IRegressionUQCalibration import I2IRegressionUQCalibrationSingleTheta, \
    I2IRegressionMultiRiskControl, ImageMiscoverageLoss, ImageCenterLowCoverageLoss
from utils.Calibration.StretchingFunctions import IdentityStretching, ExponentialStretching
from utils.DepthModel.DepthQR import DepthQR
from utils.DepthModel.Leres.lib.models.multi_depth_model_auxiv2 import RelDepthModel
from utils.DepthModel.Leres.lib.utils.net_tools import load_model_ckpt
from utils.I2IRegressionUQModel import BaselineUQModel, ResidualMagnitudeRegression, PreviousResidualUQModel, \
    PreviousResiduasWithFlowlUQModel
from utils.I2ITSModel import I2ITSModel
from utils.Calibration.TSCalibration import TSCalibration

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def parse_args_utils(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device

    assert args.ds_type.lower() in ['real', 'syn', 'synthetic'], "ds_type must be either 'real' or 'synthetic'"
    args.is_real_data = args.ds_type.lower() == 'real'
    args.suppress_plots = args.suppress_plots == 1
    if args.seed != 0:
        args.suppress_plots = True
    args.use_best_hyperparams = False
    args.apply_calibration = args.apply_calibration == 1
    if args.use_best_hyperparams:
        args = get_best_hyperparams_from_args(args)
    args.y_dim = 1
    args.backward_size = 1
    args.cal_split = False
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='quantile level')

    parser.add_argument('--dataset_name', type=str, default='KITTI',
                        help='dataset to use')

    parser.add_argument('--ds_type', type=str, default='real',
                        help='dataset type')

    parser.add_argument('--suppress_plots', type=int, default=0,
                        help='1 to disable all plots, or 0 to allow plots')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of test set size')
    parser.add_argument('--calibration_starting_update_index', type=int, default=5000,
                        help="")
    parser.add_argument('--apply_calibration', type=int, default=1,
                        help="")
    parser.add_argument('--calibration_size', type=int, default=300,
                        help="")
    parser.add_argument('--uq_method', type=str, default="previous_residual_with_flow",
                        help="")
    parser.add_argument('--backbone', type=str, default="res101",
                        help="")
    parser.add_argument('--option', type=int, default=-1,
                        help="")
    args = parser.parse_args()

    args = parse_args_utils(args)

    return args


def get_i2i_uq_model(pre_trained_network_without_last_layer, model_out_channels, device, args):
    desired_coverage_level = 1 - args.alpha
    model_params = {'pre_trained_network_without_last_layer': pre_trained_network_without_last_layer,
                    'model_out_channels': model_out_channels,
                    'device': device,
                    'loss_batch_size': 4096
                    }
    if args.uq_method == 'baseline':
        return BaselineUQModel(desired_coverage_level, **model_params)
    elif args.uq_method == 'residual_magnitude':
        return ResidualMagnitudeRegression(desired_coverage_level, **model_params)
    elif args.uq_method == 'previous_residual':
        return PreviousResidualUQModel(desired_coverage_level, **model_params)
    elif args.uq_method == 'previous_residual_with_flow':
        return PreviousResiduasWithFlowlUQModel(desired_coverage_level, **model_params)
    else:
        raise NotImplementedError("invalid uq method: ", args.uq_method)


def get_model(args, dataset):
    device = args.device
    args.method_type = 'i2i'
    trained_model_path = f"saved_models/Leres/{args.backbone}.pth"
    network = RelDepthModel(device, backbone=args.backbone)
    load_model_ckpt(argparse.Namespace(load_ckpt=trained_model_path, resume=True), network, None, None)
    network = network.depth_model
    modules = list(network.decoder_modules.outconv.adapt_conv.children())[:-2]
    network.decoder_modules.outconv.adapt_conv = torch.nn.Sequential(*modules)
    uq_model = get_i2i_uq_model(network, network.decoder_modules.midchannels[0] // 2, device, args)
    model = DepthQR(trained_model_path=trained_model_path, uq_model=uq_model,
                    device=device, tau=args.alpha, args=args, dataset=dataset, backbone=args.backbone)

    model.initialize_scalers(dataset.x_train, dataset.y_train)

    return model


def fit_predict_online(model, dataset, desired_coverage_level, calibrations: List[TSCalibration], args, data_generator):
    idx_to_start_storing = dataset.starting_test_time - dataset.val_size if isinstance(dataset, LazyDepthDataSet) else 0
    uncalibrated_train_intervals, calibrated_train_intervals, = model.predict_test_online(
        None,
        None,
        dataset.x_train, dataset.y_train, desired_coverage_level,
        args.backward_size,
        args=args, save_new_model=True, calibrations=calibrations,
        fit_on_train_set=False,
        train_idx=[], test_idx=list(range(dataset.starting_test_time)),
        store_intervals=True,
        is_train=True,
        idx_to_start_storing=idx_to_start_storing
    )
    if args.is_real_data:
        data_params = 'real/'
    else:
        data_params = 'syn/'
    data_params += args.dataset_name
    base_save_dir = 'results'
    result_per_calibration = {}
    for calibration in calibrations:
        args.calibration_method = Hyperparameters.get_cal_method(calibration.name(), args.cal_split)
        args.gamma = calibration.gamma
        args_txt = args_to_txt(args, is_calibrated=True)
        calibrated_preds = calibrated_train_intervals[calibration]
        figures_save_dir = f'{base_save_dir}/validation/figures/{data_params}/{args_txt}'
        results_info_save_dir = f'{base_save_dir}/validation/{data_params}/{args_txt}'
        res = save_performance_metrics(dataset.y_val.squeeze(),
                                       calibrated_preds,
                                       args, figures_save_dir,
                                       results_info_save_dir,
                                       initial_time=dataset.starting_test_time - 1000,
                                       calibration=calibration,
                                       dataset=dataset,
                                       is_validation=True, results=None, store_results=True)
        result_per_calibration[calibration] = res

    del uncalibrated_train_intervals
    del calibrated_train_intervals
    uncalibrated_train_intervals, calibrated_train_intervals = None, None
    uncalibrated_test_intervals, calibrated_test_intervals = model.predict_test_online(dataset.x_train, dataset.y_train,
                                                                                       dataset.x_test, dataset.y_test,
                                                                                       desired_coverage_level,
                                                                                       args.backward_size,
                                                                                       calibrations=calibrations,
                                                                                       args=args,
                                                                                       fit_on_train_set=True,
                                                                                       train_idx=list(range(
                                                                                           dataset.starting_test_time)),
                                                                                       test_idx=list(range(
                                                                                           dataset.starting_test_time,
                                                                                           dataset.data_size)),
                                                                                       store_intervals=True,
                                                                                       is_train=False,
                                                                                       )
    return uncalibrated_train_intervals, calibrated_train_intervals, uncalibrated_test_intervals, \
           calibrated_test_intervals, result_per_calibration


def get_calibration_schemes(desired_coverage_level, model, args):
    if isinstance(model, I2ITSModel):
        return get_i2i_calibration_schemes(desired_coverage_level, model, args)
    else:
        raise Exception("invalid model type for calibration")


FULL_GAMMA_SET = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 10]
SMALL_GAMMA_SET = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1, 0.5, 2]


# SMALL_GAMMA_SET =FULL_GAMMA_SET= [0.01]

def get_i2i_calibration_schemes(desired_coverage_level, model, args):
    calibrations = []

    identity_stretching = IdentityStretching()
    exp_stretching = ExponentialStretching()

    calibration_params = {'calibration_size': args.calibration_size,
                          'calibration_starting_update_index': args.calibration_starting_update_index,
                          'model': model,
                          'args': args}

    max_agg = lambda l: max(l)
    mean_agg = lambda l: np.mean(l)
    stretching = exp_stretching
    if args.option != -1:
        gammas_to_check = SMALL_GAMMA_SET
        assert 0 <= args.option < len(gammas_to_check)
        gamma1 = gammas_to_check[args.option]
        for gamma2 in gammas_to_check:
            gammas = [gamma1, gamma2]
            calibrations += [
                I2IRegressionMultiRiskControl(desired_coverage_level, gammas, **calibration_params,
                                              losses=[ImageMiscoverageLoss(args.alpha),
                                                      ImageCenterLowCoverageLoss(0.1),
                                                      ],
                                              stretching_function=stretching,
                                              aggregation=max_agg,
                                              aggregation_name='max'),

                I2IRegressionMultiRiskControl(desired_coverage_level, gammas, **calibration_params,
                                              losses=[ImageMiscoverageLoss(args.alpha),
                                                      ImageCenterLowCoverageLoss(0.1)],
                                              stretching_function=stretching,
                                              aggregation=mean_agg,
                                              aggregation_name='mean'),
            ]
    else:
        for gamma1 in FULL_GAMMA_SET:
            calibrations += [
                # I2IRegressionUQCalibrationSingleTheta(desired_coverage_level, gamma1, **calibration_params,
                #                                       stretching_function=identity_stretching),
                I2IRegressionUQCalibrationSingleTheta(desired_coverage_level, gamma1, **calibration_params,
                                                      stretching_function=exp_stretching),
            ]

    print(f"number of calibrations: {len(calibrations)}")
    return calibrations


def main():
    args = parse_args()
    set_seeds(args.seed)
    desired_coverage_level = 1 - args.alpha
    is_real_data = args.is_real_data

    device = args.device
    print("device: ", device)

    data_generator = DataGeneratorFactory.get_data_generator(args.dataset_name, is_real_data, args)

    T = min(10000, data_generator.max_data_size)

    dataset = LazyDepthDataSet(data_generator, device, T, args.test_ratio, args.calibration_starting_update_index)
    model = get_model(args, dataset)
    calibrations = get_calibration_schemes(desired_coverage_level, model, args)
    uncalibrated_train_intervals, calibrated_train_intervals, uncalibrated_test_intervals, calibrated_test_intervals, result_per_calibration = \
        fit_predict_online(model, dataset, desired_coverage_level, calibrations=calibrations,
                           args=args, data_generator=data_generator)

    if not args.suppress_plots:
        model.plot_losses()

    if is_real_data:
        data_params = 'real/'
    else:
        data_params = 'syn/'
    data_params += args.dataset_name
    args_txt = args_to_txt(args, is_calibrated=False)

    base_save_dir = 'results'

    figures_save_dir = f'{base_save_dir}/test/figures/{data_params}/{args_txt}'
    results_info_save_dir = f'{base_save_dir}/test/{data_params}/{args_txt}'
    save_performance_metrics(dataset.y_test.squeeze(),
                             uncalibrated_test_intervals, args,
                             figures_save_dir, results_info_save_dir,
                             initial_time=dataset.starting_test_time,
                             calibration=None,
                             dataset=dataset)

    if args.calibration_starting_update_index > dataset.starting_test_time:
        return

    for calibration in calibrations:
        if result_per_calibration is not None and calibration in result_per_calibration:
            res = result_per_calibration[calibration]
        else:
            res = None
        args.calibration_method = Hyperparameters.get_cal_method(calibration.name(), args.cal_split)
        args.gamma = calibration.gamma
        args_txt = args_to_txt(args, is_calibrated=True)
        calibrated_preds = calibrated_test_intervals[calibration]
        figures_save_dir = f'{base_save_dir}/test/figures/{data_params}/{args_txt}'
        results_info_save_dir = f'{base_save_dir}/test/{data_params}/{args_txt}'
        save_performance_metrics(
            dataset.y_test.squeeze(),
            calibrated_preds,
            args, figures_save_dir,
            results_info_save_dir,
            initial_time=dataset.starting_test_time,
            calibration=calibration,
            dataset=dataset,
            results=res)


if __name__ == '__main__':
    main()
