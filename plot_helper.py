import traceback

import cv2
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import circulant
import seaborn as sns

from RCOL_datasets.datasets import I2IDataset
from helper import create_folder_if_it_doesnt_exist
from utils.Calibration.ACICalibration import ACICalibration


def display_plot(x_label=None, y_label=None, title=None, display_legend=False, save_path=None, dpi=300):
    if display_legend:
        plt.legend()

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    plt.show()


def plot_syn_data_intervals(y_test, y_samples, upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                            initial_time, T, initial_plotting_time, save_dir):
    times = list(range(initial_time + initial_plotting_time, initial_time + initial_plotting_time + T))

    if y_samples is not None:
        plt.scatter(times, y_samples[0][initial_plotting_time:initial_plotting_time + T].cpu(), color='blue')

        times_rep = torch.Tensor(times).unsqueeze(0).repeat(100, 1).flatten()
        plt.scatter(times_rep, y_samples[:100, initial_plotting_time:initial_plotting_time + T].cpu(), color='blue')
        #
        # for i in range(1, 100):
        #     plt.scatter(times, y_samples[i][initial_plotting_time:initial_plotting_time + T].cpu(), color='blue')

    plt.plot(times, y_test[initial_plotting_time:initial_plotting_time + T].cpu(), color='green')

    if true_upper_q is not None and true_lower_q is not None:
        plt.plot(times, true_upper_q[initial_plotting_time:initial_plotting_time + T].cpu(), color='red', linewidth=4,
                 label='True quantiles')
        plt.plot(times, true_lower_q[initial_plotting_time:initial_plotting_time + T].cpu(), color='red', linewidth=4)

    plt.plot(times, lower_q_pred[initial_plotting_time:initial_plotting_time + T].cpu(), color='purple',
             label='Estimated quantiles', linewidth=4)
    plt.plot(times, upper_q_pred[initial_plotting_time:initial_plotting_time + T].cpu(), color='purple', linewidth=4)

    save_file_name = f"intervals_initial_plotting_time={initial_time + initial_plotting_time}_T={T}.png"
    display_plot(x_label="Time", y_label="Y", display_legend=True,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_syn_data_interval_length(upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                                  initial_time, T, initial_plotting_time, save_dir):
    times = list(range(initial_time + initial_plotting_time, initial_time + initial_plotting_time + T))

    if true_upper_q is not None and true_lower_q is not None:
        plt.plot(times, (true_upper_q[:T] - true_lower_q[:T]).cpu(), color='red', linewidth=4, label='True quantiles',
                 alpha=0.5)
    plt.plot(times, (upper_q_pred[:T] - lower_q_pred[:T]).cpu(), color='purple', linewidth=4,
             label='Estimated quantiles',
             alpha=0.5)
    save_file_name = f"intervals_length_initial_plotting_time={initial_time + initial_plotting_time}_T={T}.png"
    matplotlib.rc('font', **{'size': 16})
    display_plot(x_label="Time", y_label="Interval's length", display_legend=True,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_syn_data_results(y_test, y_samples, upper_q_pred, lower_q_pred, true_upper_q, true_lower_q, initial_time,
                          save_dir, args, calibration: ACICalibration = None):
    if args.suppress_plots:
        return
    create_folder_if_it_doesnt_exist(save_dir)

    for T in [min(1000, len(y_test)), min(50, len(y_test)), len(y_test)]:
        for initial_plotting_time in [0, 400, len(y_test) - 1050]:
            if len(y_test) < initial_plotting_time + T or initial_plotting_time < 0:
                continue
            try:
                if calibration is not None:
                    calibration.plot_parameter_vs_time(initial_time + initial_plotting_time,
                                                       initial_time + initial_plotting_time + T, save_dir=save_dir)

                plot_syn_data_intervals(y_test, y_samples, upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                                        initial_time, T, initial_plotting_time, save_dir)
                plot_syn_data_interval_length(upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                                              initial_time, T, initial_plotting_time, save_dir)
            except Exception as e:
                print(f"failed plotting because: {e}")
            traceback.print_exc()

    lengths = (upper_q_pred - lower_q_pred).cpu().numpy()
    plt.hist(lengths, bins=lengths.shape[0] // 100)
    save_file_name = f"interval_length_histogram.png"
    display_plot(x_label="Interval's length", y_label="Count", display_legend=False,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_local_coverage(coverage, local_coverage, initial_time, save_dir):
    matplotlib.rc('font', **{'size': 21})
    plt.figure(figsize=(7, 4.5))
    times = list(range(initial_time, initial_time + local_coverage.shape[0]))
    plt.plot(times, local_coverage)
    plt.axhline(coverage.float().mean().item(), ls='--')
    save_file_name = f"local coverage level.png"
    display_plot(x_label="Time", y_label="Local Coverage Level", display_legend=False,
                 save_path=f'{save_dir}/{save_file_name}')



def plot_image_lengths(lengths, save_dir, initial_time, args):
    if args.suppress_plots:
        return
    create_folder_if_it_doesnt_exist(save_dir)
    plt.plot(list(range(initial_time, initial_time + len(lengths))), lengths)
    save_file_name = f"image_lengths.png"
    display_plot(x_label="Time", y_label="Length", display_legend=False,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_image_coverages(image_coverages, save_dir, initial_time, args):
    if args.suppress_plots:
        return
    create_folder_if_it_doesnt_exist(save_dir)
    plt.plot(list(range(initial_time, initial_time + len(image_coverages))), image_coverages)
    save_file_name = f"image_coverages.png"
    display_plot(x_label="Time", y_label="Coverage", display_legend=False,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_center_image_coverages(image_center_coverages, save_dir, initial_time, args):
    if args.suppress_plots:
        return
    create_folder_if_it_doesnt_exist(save_dir)
    plt.plot(list(range(initial_time, initial_time + len(image_center_coverages))), image_center_coverages)
    save_file_name = f"image_center_coverages.png"
    display_plot(x_label="Time", y_label="Center coverage", display_legend=False,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_image_uq(lower_q_pred, upper_q_pred, dataset: I2IDataset, save_dir, initial_time,is_validation, args):
    if args.suppress_plots:
        return
    if is_validation:
        scaling_factor = dataset.val_scaling_factors.cpu().squeeze()
    else:
        scaling_factor = dataset.test_scaling_factors.cpu().squeeze()
    lower_q_pred = lower_q_pred.squeeze().cpu()
    upper_q_pred = upper_q_pred.squeeze().cpu()
    length = upper_q_pred - lower_q_pred
    create_folder_if_it_doesnt_exist(save_dir)
    skip = 200 if is_validation else 10
    for i in range(0, lower_q_pred.shape[0], skip):
        save_path = f"{save_dir}/intervals_lengths_time={i + initial_time}.png"
        im = (length[i] / scaling_factor[i]).squeeze()
        cv2.imwrite(save_path, im.numpy().astype(np.uint16))

def plot_image_lower_q(lower_q_pred, upper_q_pred, dataset: I2IDataset, save_dir, initial_time,is_validation, args):
    if args.suppress_plots:
        return
    if is_validation:
        scaling_factor = dataset.val_scaling_factors.cpu().squeeze()
    else:
        scaling_factor = dataset.test_scaling_factors.cpu().squeeze()
    lower_q_pred = lower_q_pred.squeeze().cpu()
    create_folder_if_it_doesnt_exist(save_dir)
    skip = 200 if is_validation else 10
    for i in range(0, lower_q_pred.shape[0], skip):
        save_path = f"{save_dir}/lower_q_time={i + initial_time}.png"
        im = (lower_q_pred[i] / scaling_factor[i]).squeeze()
        cv2.imwrite(save_path, im.numpy().astype(np.uint16))


def plot_image_upper_q(lower_q_pred, upper_q_pred, dataset: I2IDataset, save_dir, initial_time,is_validation, args):
    if args.suppress_plots:
        return
    if is_validation:
        scaling_factor = dataset.val_scaling_factors.cpu().squeeze()
    else:
        scaling_factor = dataset.test_scaling_factors.cpu().squeeze()
    create_folder_if_it_doesnt_exist(save_dir)
    upper_q_pred = upper_q_pred.squeeze().cpu()
    skip = 200 if is_validation else 10
    for i in range(0, upper_q_pred.shape[0], skip):
        save_path = f"{save_dir}/lower_q_time={i + initial_time}.png"
        im = (upper_q_pred[i] / scaling_factor[i]).squeeze()
        cv2.imwrite(save_path, im.numpy().astype(np.uint16))


def plot_image_estimated_depth(estimated_means, dataset: I2IDataset, save_dir, initial_time, is_validation, args):
    if args.suppress_plots:
        return
    if is_validation:
        scaling_factor = dataset.val_scaling_factors.cpu().squeeze()
    else:
        scaling_factor = dataset.test_scaling_factors.cpu().squeeze()
    estimated_means = estimated_means.squeeze().cpu()
    create_folder_if_it_doesnt_exist(save_dir)
    skip = 200 if is_validation else 10
    for i in range(0, estimated_means.shape[0], skip):
        save_path = f"{save_dir}/estimated_depth_time={i + initial_time}.png"
        im = (estimated_means[i] / scaling_factor[i]).squeeze()
        cv2.imwrite(save_path, im.numpy().astype(np.uint16))
