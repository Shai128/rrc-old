import pickle

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.linalg import circulant

import helper
from RCOL_datasets.datasets import SynDataGenerator, DataGenerator, WindowSynDataGenerator
from helper import create_folder_if_it_doesnt_exist
from hyperparameters import Hyperparameters
from plot_helper import plot_image_uq, plot_image_estimated_depth, plot_image_lower_q, plot_image_upper_q
import traceback
from utils.TSQR import batch_pinball_loss
from utils.UncertaintyQuantificationResults.I2IRegressionUQ import I2IRegressionUQSet


def get_avg_miscoverage_streak_len(y, upper_q, lower_q):
    coverages = (y <= upper_q) & (y >= lower_q)
    streaks_lengths = []
    last_was_covered = True
    for i in range(len(coverages)):
        if coverages[i]:
            last_was_covered = True
            continue
        if last_was_covered:
            streaks_lengths.append(1)
        else:
            streaks_lengths[-1] += 1
        last_was_covered = False

    return np.mean(streaks_lengths)


def args_to_txt(args, is_calibrated=False):
    hp = Hyperparameters.from_args(args, is_calibrated)
    hp.is_calibrated = is_calibrated
    return hp.to_folder_name()


def add_depth_results_into_results_dict(results, y, upper_q, lower_q, title_prefix, args, dataset, is_validation):
    if is_validation:
        valid_idx_mask = dataset.val_valid_idx_mask.cpu()
        image_center = dataset.val_image_center.cpu()
        scaling_factor = dataset.val_scaling_factors.cpu()
        meters_factor = dataset.val_meters_factors.cpu()
    else:
        valid_idx_mask = dataset.test_valid_idx_mask.cpu()
        image_center = dataset.test_image_center.cpu()
        scaling_factor = dataset.test_scaling_factors.cpu()
        meters_factor = dataset.test_meters_factors.cpu()

    try:
        original_y = y.cpu()
        original_upper_q = upper_q.cpu()
        original_lower_q = lower_q.cpu()
        original_valid_idx_mask = valid_idx_mask.cpu()
        result_arrays = {}
        for image_part in ['image', 'center', 'non center']:
            coverages = []
            lengths = []
            pb_losses = []
            pixelwise_coverage = torch.zeros_like(original_y[0].squeeze()).cpu()
            pixelwise_n_valid = torch.zeros_like(original_y[0].squeeze()).cpu()
            avg_unscaled_y = []
            max_unscaled_y = []
            for i in range(0, original_y.shape[0]):
                valid_idx_mask = original_valid_idx_mask[i].to(y.device).squeeze()
                center_pixel = image_center[i]
                if image_part == 'image':
                    relevant_idx = torch.ones_like(valid_idx_mask)
                elif image_part == 'center':
                    relevant_idx = torch.zeros_like(valid_idx_mask)
                    relevant_idx[center_pixel[0] - 25: center_pixel[0] + 25,
                    center_pixel[1] - 25: center_pixel[1] + 25] = True
                elif image_part == 'non center':
                    relevant_idx = torch.ones_like(valid_idx_mask, dtype=torch.bool)
                    relevant_idx[center_pixel[0] - 25: center_pixel[0] + 25,
                    center_pixel[1] - 25: center_pixel[1] + 25] = False
                else:
                    print(f"Error. no relevant_idx was defined for image_part={image_part}. setting the default one")
                    relevant_idx = torch.ones_like(valid_idx_mask)
                relevant_idx = relevant_idx & valid_idx_mask
                y = original_y[i].squeeze()
                upper_q = original_upper_q[i].squeeze()
                lower_q = original_lower_q[i].squeeze()

                v = ((y <= upper_q) & (y >= lower_q)).float() * relevant_idx.float()
                pixelwise_coverage += v.cpu().squeeze()
                pixelwise_n_valid += relevant_idx

                y = y[relevant_idx].squeeze()
                upper_q = upper_q[relevant_idx].squeeze()
                lower_q = lower_q[relevant_idx].squeeze()

                coverages += [((y <= upper_q) & (y >= lower_q)).float().mean().item()]
                lengths += [(meters_factor[i] * (upper_q[i] - lower_q[i]).mean() / scaling_factor[i])]

                lower_pb = batch_pinball_loss(args.alpha / 2, lower_q, y)
                upper_pb = batch_pinball_loss(1 - args.alpha / 2, upper_q, y)
                pb_loss = (lower_pb + upper_pb) / 2
                pb_losses += [pb_loss.item()]

                try:
                    avg_unscaled_y += [(meters_factor[i] * y.mean() / scaling_factor[i]).item()]
                    max_unscaled_y += [(meters_factor[i] * y.max() / scaling_factor[i]).item()]
                except:
                    traceback.print_exc()
                    print("failed calculating max unscaled y")

            coverages = np.array(coverages) * 100
            delta_coverage = np.mean(np.abs(coverages - (1 - args.alpha) * 100))
            actual_delta_coverage = np.mean(np.abs(coverages - np.mean(coverages)))
            max_delta_coverage = np.max(np.abs(coverages - (1 - args.alpha) * 100))
            max_actual_delta_coverage = np.max(np.abs(coverages - np.mean(coverages)))

            avg_pixelwise_coverage = (pixelwise_coverage / pixelwise_n_valid)[pixelwise_n_valid > 50]
            if len(avg_pixelwise_coverage) == 0:
                avg_pixelwise_coverage = (pixelwise_coverage / pixelwise_n_valid)

            pixelwise_delta_coverage = (avg_pixelwise_coverage - (1 - args.alpha)).abs().mean().item() * 100
            pixelwise_actual_delta_coverage = (avg_pixelwise_coverage - avg_pixelwise_coverage.mean()).abs().mean().item() * 100
            pixelwise_max_delta_coverage = (avg_pixelwise_coverage - (1 - args.alpha)).abs().max().item() * 100
            pixelwise_max_actual_delta_coverage = (avg_pixelwise_coverage - avg_pixelwise_coverage.mean()).abs().max().item() * 100

            results[f'{title_prefix} {image_part} pb loss'] = np.mean(pb_losses)
            results[f'{title_prefix} {image_part} coverage'] = np.mean(coverages)
            results[f'{title_prefix} {image_part} average length'] = np.mean(lengths)
            results[f'{title_prefix} {image_part} median length'] = np.median(lengths)
            results[f'{title_prefix} {image_part} delta coverage'] = delta_coverage
            results[f'{title_prefix} {image_part} actual delta coverage'] = actual_delta_coverage
            results[f'{title_prefix} {image_part} low coverage occurrences0(%)'] = (np.array(
                coverages) < 100 * 0.4).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences1(%)'] = (np.array(
                coverages) < 100 * 0.5).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences2(%)'] = (np.array(
                coverages) < 100 * 0.6).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences3(%)'] = (np.array(
                coverages) < 100 * 0.65).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences4(%)'] = (np.array(
                coverages) < 100 * 0.7).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences5(%)'] = (np.array(
                coverages) < 100 * 0.75).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences5.5(%)'] = (np.array(
                coverages) < 100 * 0.78).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences5.8(%)'] = (np.array(
                coverages) < 100 * 0.79).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} low coverage occurrences6(%)'] = (np.array(
                coverages) < 100 * 0.8).astype(
                np.float).mean().item() * 100
            results[f'{title_prefix} {image_part} max delta coverage'] = max_delta_coverage
            results[f'{title_prefix} {image_part} max actual delta coverage'] = max_actual_delta_coverage
            results[f'{title_prefix} {image_part} pixel-wise Δ-coverage'] = pixelwise_delta_coverage
            results[f'{title_prefix} {image_part} max pixel-wise Δ-coverage'] = pixelwise_max_delta_coverage
            results[f'{title_prefix} {image_part} actual pixel-wise Δ-coverage'] = pixelwise_actual_delta_coverage
            results[
                f'{title_prefix} {image_part} actual max pixel-wise Δ-coverage'] = pixelwise_max_actual_delta_coverage
            results[f'{title_prefix} {image_part} max depth'] = np.max(max_unscaled_y)
            results[f'{title_prefix} {image_part} average depth'] = np.mean(avg_unscaled_y)

            result_arrays[f'{image_part} coverages'] = coverages
            result_arrays[f'{image_part} lengths'] = lengths
        return result_arrays
    except:
        print("error in computing depth results")
        traceback.print_exc()
        return {}


def depth_data_save_performance_metrics(y_test: torch.Tensor, prediction_intervals: I2IRegressionUQSet, args,
                                        figures_save_dir,
                                        results_info_save_dir, initial_time, calibration,
                                        dataset, is_validation=False, results=None, store_results=True):
    intervals = prediction_intervals.unscaled_intervals
    estimated_means = prediction_intervals.unscaled_mean
    lower_q_pred, upper_q_pred = intervals[..., 0], intervals[..., 1]
    if results is None:
        results = {}
    # Marginal results
    if is_validation:
        title_prefix = "Estimated validation quantiles"
    else:
        title_prefix = "Estimated quantiles"

    result_arrays = add_depth_results_into_results_dict(results, y_test, upper_q_pred, lower_q_pred,
                                                        title_prefix=title_prefix, args=args, dataset=dataset,
                                                        is_validation=is_validation)
    coverages = result_arrays['image coverages']
    lengths = result_arrays['image lengths']
    center_coverages = result_arrays['center coverages']
    center_lengths = result_arrays['center lengths']
    helper.create_folder_if_it_doesnt_exist(figures_save_dir)
    if not args.suppress_plots:
        if is_validation:
            with open(f"{figures_save_dir}/val_augmentation_info.pkl", 'wb') as f:
                pickle.dump(dataset.val_augmentation_info, f)
        else:
            with open(f"{figures_save_dir}/test_augmentation_info.pkl", 'wb') as f:
                pickle.dump(dataset.test_augmentation_info, f)
        with open(f"{figures_save_dir}/coverages.pkl", 'wb') as f:
            pickle.dump(coverages, f)
        with open(f"{figures_save_dir}/center_coverages.pkl", 'wb') as f:
            pickle.dump(center_coverages, f)
        with open(f"{figures_save_dir}/lengths.pkl", 'wb') as f:
            pickle.dump(lengths, f)
        with open(f"{figures_save_dir}/center_lengths.pkl", 'wb') as f:
            pickle.dump(center_lengths, f)

    if not is_validation:
        plot_image_uq(lower_q_pred, upper_q_pred, dataset, f"{figures_save_dir}/uq", initial_time, is_validation, args)
        plot_image_lower_q(lower_q_pred, upper_q_pred, dataset, f'{figures_save_dir}/lower_q', initial_time,
                           is_validation, args)
        plot_image_upper_q(lower_q_pred, upper_q_pred, dataset, f'{figures_save_dir}/upper', initial_time,
                           is_validation, args)
        if calibration is None and estimated_means is not None:
            plot_image_estimated_depth(estimated_means, dataset, f"{figures_save_dir}/estimated_depth", initial_time,
                                       is_validation, args)

    if store_results:
        if calibration is not None:
            calibration.plot_parameter_vs_time(initial_time,
                                               initial_time + len(coverages), save_dir=figures_save_dir)
        create_folder_if_it_doesnt_exist(results_info_save_dir)
        pd.DataFrame(results, index=[args.seed]).to_csv(f"{results_info_save_dir}/seed={args.seed}.csv")
    return results


def save_performance_metrics(y_test,
                             uncertainty_estimation_set: I2IRegressionUQSet, args, figures_save_dir,
                             results_info_save_dir, initial_time, calibration=None, dataset=None,
                             is_validation=False, results=None, store_results=True):
    return depth_data_save_performance_metrics(y_test=y_test, prediction_intervals=uncertainty_estimation_set,
                                               args=args,
                                               figures_save_dir=figures_save_dir,
                                               results_info_save_dir=results_info_save_dir, initial_time=initial_time,
                                               calibration=calibration, dataset=dataset, is_validation=is_validation,
                                               results=results, store_results=store_results)
