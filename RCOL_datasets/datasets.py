"""
Part of the code is taken from https://github.com/yromano/cqr
"""
import abc
from abc import ABC

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys

import helper
from helper import set_seeds, get_current_seed
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA

from utils.DataScaler import DataScaler
import argparse

from utils.DepthModel.Leres.data.multi_dataset import MultiDataset

sys.path.insert(1, '..')
DEPTH_DATASETS = ['diversedepth', 'kitti']


class DataGeneratorFactory:
    possible_syn_dataset_names = ['simple_invertible_data', 'simple_non_invertible_data', 'simple_continuous_data',
                                  'indp_x_continuous_data', 'simple_cyclic_data',
                                  'miscoverage_streak', 'asymmetric_miscoverage_streak',
                                  'window', 'nox']

    @staticmethod
    def get_data_generator(dataset_name, is_real, args):
        if is_real:
            return DataGeneratorFactory.get_real_data_generator(dataset_name, args)
        else:
            return DataGeneratorFactory.get_syn_data_generator(dataset_name, args)

    @staticmethod
    def get_real_data_generator(dataset_name, args):
        if 'y_dim' in dataset_name:
            data_y_dim = int(re.search(r'\d+', re.search(r'y_dim_\d+', dataset_name).group()).group())
        else:
            data_y_dim = 1
        if dataset_name.lower() in DEPTH_DATASETS:
            return DepthDataGenerator(dataset_name, args, data_y_dim, 10000)
        return RealDataGenerator(dataset_name, args, data_y_dim)

    @staticmethod
    def get_syn_data_generator(dataset_name, args):

        assert any(
            possible_dataset in dataset_name for possible_dataset in DataGeneratorFactory.possible_syn_dataset_names)
        if 'x_dim' in dataset_name:
            x_dim = int(re.search(r'\d+', re.search(r'x_dim_\d+', dataset_name).group()).group())
        else:
            x_dim = 1
        if 'z_dim' in dataset_name:
            data_z_dim = int(re.search(r'\d+', re.search(r'z_dim_\d+', dataset_name).group()).group())
        else:
            data_z_dim = 1

        if 'window' in dataset_name:
            window_length = float(re.search(r'\d+', re.search(r'len_\d+', dataset_name).group()).group())
            return WindowSynDataGenerator(x_dim, data_z_dim, window_length)
        elif 'nox' in dataset_name:
            return NoXSynDataGenerator()
        else:
            assert False


class DataGenerator(ABC):
    @abc.abstractmethod
    def __init__(self, args, y_dim):
        self.args = args
        self.y_dim = y_dim

    @abc.abstractmethod
    def undo_n_steps(self, data_info, n):
        pass

    @abc.abstractmethod
    def __generate_data_aux(self, T, x=None, previous_data_info=None, **kwargs):
        pass

    def generate_data(self, T, x=None, previous_data_info=None, **kwargs):
        x, y, curr_data_info = self.__generate_data_aux(T + self.y_dim - 1, x, previous_data_info, **kwargs)
        if self.y_dim > 1:
            y_mat = torch.zeros(T, self.y_dim, device=y.device)
            for i in range(T):
                y_mat[i] = y[i:i + self.y_dim]
            y = y_mat
            x = x[:T]
            curr_data_info = self.undo_n_steps(curr_data_info, n=self.y_dim - 1)
        return x, y, curr_data_info


def get_features_to_use_as_y(dataset_name) -> list:
    if dataset_name == 'tetuan_power':
        return [0, 1]
    else:
        raise NotImplementedError()


class RealDataGenerator(DataGenerator):
    def __init__(self, dataset_name, args, y_dim):
        super().__init__(args, y_dim)
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self):
        self.x, self.y = GetDataset(self.dataset_name, 'datasets/real_data/')
        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y)
        self.max_data_size = self.x.shape[0]

    def _DataGenerator__generate_data_aux(self, T, x=None, previous_data_info=None, device='cpu'):

        if previous_data_info is None:
            starting_time = 0
        else:
            starting_time = previous_data_info['ending_time'] + 1

        current_process_info = {'ending_time': starting_time + T - 1}
        return self.x[starting_time: starting_time + T].cpu().to(device), \
               self.y[starting_time:starting_time + T].cpu().to(device), current_process_info

    def undo_n_steps(self, data_info, n):
        data_info = {'ending_time': data_info['ending_time'] - n}
        return data_info


class NoXSynDataGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.x = torch.zeros(20000, 1)
        y_list = []
        for i in range(10):
            y_list += [torch.randn(1000, 1), torch.randn(1000, 1) * 10]
        self.y = torch.cat(y_list, dim=0)
        self.max_data_size = self.x.shape[0]

    def generate_data(self, T, x=None, previous_data_info=None, device='cpu'):
        if previous_data_info is None:
            starting_time = 0
        else:
            starting_time = previous_data_info['ending_time'] + 1

        current_process_info = {'ending_time': starting_time + T - 1}
        return self.x[starting_time: starting_time + T].cpu().to(device), \
               self.y[starting_time:starting_time + T].cpu().to(device), current_process_info


class SynDataGenerator(ABC):
    def __init__(self):
        self.max_data_size = np.inf

    @abc.abstractmethod
    def generate_data(self, T, x=None, previous_data_info=None, n_samples=1, device='cpu',
                      current_process_info=None, use_constant_seed=True):
        pass

    @abc.abstractmethod
    def get_y_given_x_and_uncertainty(self, x, uncertainty, previous_data_info=None):
        pass

    @abc.abstractmethod
    def get_oracle_quantiles(self, x_test, alpha, previous_data_info=None, current_process_info=None):
        pass


class SimpleSynDataGenerator(SynDataGenerator):
    def __init__(self, x_dim, data_z_dim, z_rho=0.9):
        super().__init__()
        self.x_dim, self.data_z_dim = x_dim, data_z_dim
        self.beta = torch.rand(x_dim)
        self.beta /= self.beta.norm(p=1)

        self.beta2 = torch.rand(data_z_dim)
        self.beta2 /= self.beta2.norm(p=1)
        self.z_rho = z_rho

    @staticmethod
    def generate_beta(dim):
        beta = torch.rand(dim)
        beta /= beta.norm(p=1)
        return beta

    @abc.abstractmethod
    def generate_y_given_x_z(self, x, z, device='cpu', previous_data_info=None, current_process_info=None):
        pass

    def generate_x(self, z, n_samples, T, device, previous_data_info=None):
        x = torch.rand(n_samples, T, self.x_dim, device=device)
        return x

    def generate_data_given_z(self, z, x=None, n_samples=1, device='cpu', previous_data_info=None,
                              current_process_info=None):
        if x is None:
            x = self.generate_x(z, n_samples, z.shape[1], device, previous_data_info=previous_data_info)
        return x, self.generate_y_given_x_z(x, z, device, previous_data_info=previous_data_info,
                                            current_process_info=current_process_info)

    def generate_z(self, T, n_samples, device, previous_data_info=None, current_process_info=None):
        rho = self.z_rho
        z = torch.zeros(n_samples, T, self.data_z_dim, device=device)
        if previous_data_info is not None:
            pre_z = previous_data_info['z']
        else:
            pre_z = None

        if pre_z is not None and current_process_info is None:
            assert len(pre_z.shape) == 3 and pre_z.shape[2] == self.data_z_dim and pre_z.shape[0] >= 1
            pre_z = pre_z[:, -1:]
            initial_new_index = len(pre_z)
            z = torch.cat([pre_z.repeat(n_samples, 1, 1), z], dim=1)
        elif pre_z is not None and current_process_info is not None:
            initial_new_index = 1
            z[:, 0] = rho * pre_z[:, - 1] + torch.randn(n_samples, self.data_z_dim, device=device) * (
                    (1 - rho ** 2) ** 0.5)
        else:
            initial_new_index = 1
            z[:, 0] = torch.randn(n_samples, self.data_z_dim, device=device)

        if current_process_info is not None:
            current_process_z = current_process_info['z'].repeat(n_samples, 1, 1)
        else:
            current_process_z = z

        for t in range(initial_new_index, z.shape[1]):
            z[:, t] = rho * current_process_z[:, t - 1] + torch.randn(n_samples, self.data_z_dim, device=device) * (
                    (1 - rho ** 2) ** 0.5)

        if pre_z is not None and current_process_info is None:
            z = z[:, pre_z.shape[0]:]
        return z

    def generate_data(self, T, x=None, previous_data_info=None, n_samples=1, device='cpu',
                      current_process_info=None, get_z=False, use_constant_seed=True):
        if use_constant_seed:
            initial_seed = get_current_seed()
            set_seeds(0)
        z = self.generate_z(T, n_samples=n_samples, previous_data_info=previous_data_info,
                            current_process_info=current_process_info, device=device)
        x, y = self.generate_data_given_z(z, x=x, n_samples=n_samples, device=device,
                                          previous_data_info=previous_data_info,
                                          current_process_info=current_process_info)
        initial_time = 0 if previous_data_info is None else previous_data_info['ending_time'] + 1
        curr_data_info = {'z': z, 'y': y, 'ending_time': initial_time + T - 1}
        if n_samples == 1:
            z, x, y = z.squeeze(0), x.squeeze(0), y.squeeze(0)

        if use_constant_seed:
            set_seeds(initial_seed)

        if get_z:
            return z, x, y, curr_data_info
        else:
            return x, y, curr_data_info

    def get_oracle_quantiles(self, x_test, alpha, previous_data_info=None, current_process_info=None):

        device = x_test.device
        n_points_to_sample = 2000
        z_set = torch.randn(n_points_to_sample, self.data_z_dim, device=device)
        z_set_size = z_set.shape[0]
        z_set = z_set.reshape(z_set_size, self.data_z_dim)

        unflattened_y_set = self.generate_y_given_x_z(x_test.unsqueeze(0).repeat(z_set_size, 1, 1),
                                                      z_set.unsqueeze(1).repeat(1, x_test.shape[0], 1), device=device,
                                                      current_process_info=current_process_info,
                                                      previous_data_info=previous_data_info)

        y_upper = unflattened_y_set.quantile(dim=0, q=1 - alpha / 2)
        y_lower = unflattened_y_set.quantile(dim=0, q=alpha / 2)
        return y_lower, y_upper

    def get_y_given_x_and_uncertainty(self, x, z, previous_data_info=None):
        return self.generate_y_given_x_z(x, z, device=x.device, previous_data_info=previous_data_info)

    def reduce_x_and_z(self, x, z):
        self.beta = self.beta.to(x.device)
        self.beta2 = self.beta2.to(x.device)
        x = x @ self.beta
        z = z @ self.beta2
        return x, z


class WindowSynDataGenerator(SimpleSynDataGenerator):

    def __init__(self, x_dim, data_z_dim, window_length):
        super().__init__(x_dim, data_z_dim, z_rho=0.)
        self.window_length = window_length
        max_data_size = 50000
        self.y_rho = 0.5
        distr_num = torch.zeros(max_data_size).int()
        curr_idx = 0
        curr_count = 0
        while curr_idx < len(distr_num) - 1:
            next_idx = curr_idx + (window_length + torch.randn(1) * min(10, window_length // 20)).int().item()
            next_idx = max(next_idx, curr_idx)
            next_idx = min(next_idx, len(distr_num))

            distr_num[curr_idx:next_idx] = curr_count

            curr_count = curr_count + 1
            curr_idx = next_idx

        self.data_type_idx = distr_num.int()
        n_data_types = self.data_type_idx.max()
        self.x_betas = [WindowSynDataGenerator.generate_beta(x_dim) for _ in range(n_data_types)]
        self.noise_levels = [(torch.rand(1).abs().item() * 10 + 20) for _ in range(n_data_types)]

    def generate_y_given_x_z(self, x, z, device='cpu', previous_data_info=None, current_process_info=None):

        x, z = x.to(device), z.to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
        initial_time = 0 if previous_data_info is None else previous_data_info['ending_time'] + 1
        assert z.shape[0] == x.shape[0] or x.shape[0] == 1 or z.shape[0] == 1
        if x.shape[0] == 1 and z.shape[0] != 1:
            x = x.repeat(z.shape[0], 1, 1)
        if x.shape[0] != 1 and z.shape[0] == 1:
            z = z.repeat(x.shape[0], 1, 1)

        if previous_data_info is not None:
            pre_y = previous_data_info['y']
            pre_y = pre_y[:, -1:].to(device).repeat(x.shape[0], 1)

        else:
            pre_y = torch.ones(x.shape[0], 1, device=device)

        y = torch.zeros(x.shape[0], x.shape[1], device=device)
        if current_process_info is None:
            pre_y_and_curr_y = torch.cat([pre_y, y], dim=1)
            for t in range(1, pre_y_and_curr_y.shape[1]):
                curr_time_factor = self.curr_time_factor(x[:, t - 1], z[:, t - 1], t - 1 + initial_time)
                pre_y_and_curr_y[:, t] = self.y_rho * pre_y_and_curr_y[:, t - 1] + curr_time_factor
            y = pre_y_and_curr_y[:, 1:]

        else:
            current_y_process = current_process_info['y'].repeat(x.shape[0], 1)
            pre_y_and_curr_y = torch.cat([pre_y, current_y_process], dim=1)
            for t in range(0, y.shape[1]):
                curr_time_factor = self.curr_time_factor(x[:, t], z[:, t], t + initial_time)
                y[:, t] = self.y_rho * pre_y_and_curr_y[:, t - 1 + pre_y.shape[1]] + curr_time_factor
        return y

    def curr_time_factor(self, x, z, absolute_time):
        reduced_x = x @ self.x_betas[self.data_type_idx[absolute_time]].to(x.device)
        reduced_x = reduced_x.abs()
        small_uncertainty = torch.sin(2 * x[..., 1] * z[..., 0]) * 2

        if int(self.data_type_idx[absolute_time].item()) % 2 == 0:
            noise_level = self.noise_levels[self.data_type_idx[absolute_time]] ** 2
        else:
            noise_level = 1

        large_uncertainty = noise_level * reduced_x
        return small_uncertainty + large_uncertainty.abs() + 1000 * z[..., 1].abs()


class MiscoverageStreaksSynDataGenerator(SimpleSynDataGenerator):

    @staticmethod
    def get_uncertainty_strength_per_time(beginning_uncertainty_strength, uncertainty_strength_factor,
                                          increasing_uncertainty_streak_legnth, small_uncertainty_streak_length):
        uncertainty_strength = torch.zeros(99999)
        desired_streak_len = max(1, int(torch.randn(1) * np.sqrt(small_uncertainty_streak_length) * 0.5 + \
                                        small_uncertainty_streak_length))
        curr_has_small_uncertainty = True
        t = 0
        while t < uncertainty_strength.shape[0]:
            if curr_has_small_uncertainty:
                uncertainty_strength[t:t + desired_streak_len] = torch.zeros(desired_streak_len)
            else:
                uncertainty_strength[t:t + desired_streak_len] = (uncertainty_strength_factor ** \
                                                                  torch.arange(desired_streak_len)) * \
                                                                 beginning_uncertainty_strength
            t += desired_streak_len
            curr_has_small_uncertainty = not curr_has_small_uncertainty
            if curr_has_small_uncertainty:
                desired_streak_len = min(max(1, int(torch.randn(1) * np.sqrt(small_uncertainty_streak_length) * 0.5 + \
                                                    small_uncertainty_streak_length)),
                                         uncertainty_strength.shape[0] - t)
            else:
                desired_streak_len = min(
                    max(1, int(torch.randn(1) * np.sqrt(increasing_uncertainty_streak_legnth) * 0.5 + \
                               increasing_uncertainty_streak_legnth)),
                    uncertainty_strength.shape[0] - t)
        return uncertainty_strength

    def __init__(self, x_dim, data_z_dim, streak_length):
        super().__init__(x_dim, data_z_dim)
        self.y_rho = 0.5
        self.increasing_uncertainty_streak_legnth = streak_length
        self.small_uncertainty_streak_length = 9 * streak_length
        self.beginning_uncertainty_strength = 30
        self.uncertainty_strength_factor = 2
        self.rho = 0
        self.uncertainty_strength = MiscoverageStreaksSynDataGenerator.get_uncertainty_strength_per_time(
            self.beginning_uncertainty_strength, self.uncertainty_strength_factor,
            self.increasing_uncertainty_streak_legnth, self.small_uncertainty_streak_length)

    def generate_y_given_x_z(self, x, z, device='cpu', previous_data_info=None, current_process_info=None):
        self.uncertainty_strength = self.uncertainty_strength.to(device)

        x, z = x.to(device), z.to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
        initial_time = 0 if previous_data_info is None else previous_data_info['ending_time'] + 1
        assert z.shape[0] == x.shape[0] or x.shape[0] == 1 or z.shape[0] == 1
        if x.shape[0] == 1 and z.shape[0] != 1:
            x = x.repeat(z.shape[0], 1, 1)
        if x.shape[0] != 1 and z.shape[0] == 1:
            z = z.repeat(x.shape[0], 1, 1)

        if previous_data_info is not None:
            pre_y = previous_data_info['y']
            pre_y = pre_y[:, -1:].to(device).repeat(x.shape[0], 1)

        else:
            pre_y = torch.ones(x.shape[0], 1, device=device)

        y = torch.zeros(x.shape[0], x.shape[1], device=device)
        if current_process_info is None:
            pre_y_and_curr_y = torch.cat([pre_y, y], dim=1)
            for t in range(1, pre_y_and_curr_y.shape[1]):
                curr_time_factor = self.curr_time_factor(x[:, t - 1], z[:, t - 1], t - 1 + initial_time)
                pre_y_and_curr_y[:, t] = self.y_rho * pre_y_and_curr_y[:, t - 1] + curr_time_factor
            y = pre_y_and_curr_y[:, 1:]

        else:
            current_y_process = current_process_info['y'].repeat(x.shape[0], 1)
            pre_y_and_curr_y = torch.cat([pre_y, current_y_process], dim=1)
            for t in range(0, y.shape[1]):
                curr_time_factor = self.curr_time_factor(x[:, t], z[:, t], t + initial_time)
                y[:, t] = self.y_rho * pre_y_and_curr_y[:, t - 1 + pre_y.shape[1]] + curr_time_factor
        return y

    def generate_x(self, z, n_samples, T, device, previous_data_info=None):
        x = torch.rand(n_samples, T, self.x_dim, device=device)
        beginning_time = 0 if previous_data_info is None else previous_data_info['ending_time'] + 1
        uncertainty_strength = self.uncertainty_strength[beginning_time:beginning_time + T]
        x[..., 0] = (uncertainty_strength > 0).float()
        return x

    def curr_time_factor(self, x, z, absolute_time):
        reduced_x, reduced_z = self.reduce_x_and_z(x, z)
        return 5 * x[..., 1] * z[..., 1] + z[..., 0] * self.uncertainty_strength[absolute_time] * reduced_x


class AsymmetricMiscoverageStreaksSynDataGenerator(MiscoverageStreaksSynDataGenerator):

    def __init__(self, x_dim, data_z_dim, streak_length):
        super().__init__(x_dim, data_z_dim, streak_length)
        self.beginning_uncertainty_strength = 10
        self.uncertainty_strength_factor = 1.3
        self.uncertainty_strength = MiscoverageStreaksSynDataGenerator.get_uncertainty_strength_per_time(
            self.beginning_uncertainty_strength, self.uncertainty_strength_factor,
            self.increasing_uncertainty_streak_legnth, self.small_uncertainty_streak_length)

    def curr_time_factor(self, x, z, absolute_time):
        reduced_x, reduced_z = self.reduce_x_and_z(x, z)
        return 2 * x[..., 1] * z[..., 1] + z[..., 0].abs() * self.uncertainty_strength[absolute_time] * reduced_x


class DepthDataGenerator(RealDataGenerator):

    def __init__(self, dataset_name, args, y_dim, max_data_size):
        self.max_data_size = max_data_size
        super().__init__(dataset_name, args, y_dim)

    def load_data(self):
        train_options = argparse.Namespace(dataroot="", phase="train", phase_anno="train_annotations_onlyvideos")
        self.depth_dataset = MultiDataset(opt=train_options, dataset_name=self.dataset_name)

    def __getitem__(self, item):
        return self.depth_dataset[item]

    def _DataGenerator__generate_data_aux(self, T, x=None, previous_data_info=None, device='cpu'):
        if previous_data_info is None:
            starting_time = 0
        else:
            starting_time = previous_data_info['ending_time'] + 1

        current_process_info = {'ending_time': starting_time + T - 1}
        return None, None, current_process_info


class DataSet:

    def __init__(self, data_generator, is_real_data, device, T, test_ratio, calibration_starting_update_index):
        self.data_generator = data_generator
        self.is_real_data = is_real_data
        self.device = device
        self.test_ratio = test_ratio
        self.generate_data(T, calibration_starting_update_index)
        self.data_size = T

    def generate_data(self, T, calibration_starting_update_index):
        validation_ratio = 0
        train_ratio = 1 - self.test_ratio - validation_ratio
        self.data_generator = self.data_generator
        self.is_real_data = self.is_real_data
        self.x_train, self.y_train, self.training_data_info = self.data_generator.generate_data(int(train_ratio * T),
                                                                                                device=self.device)
        if len(self.y_train.shape) == 1:
            self.y_train = self.y_train.unsqueeze(-1)

        self.data_scaler = DataScaler()
        self.data_scaler.initialize_scalers(
            self.x_train[:calibration_starting_update_index],
            self.y_train[:calibration_starting_update_index]
        )
        # self.y_train = self.data_scaler.scale_y(self.y_train)

        if validation_ratio is not None and validation_ratio > 0:
            self.x_val, self.y_val, self.pre_test_data_info = self.data_generator.generate_data(
                int(validation_ratio * T),
                device=self.device,
                previous_data_info=self.training_data_info)
            self.y_val = self.y_val.unsqueeze(1)
            self.starting_test_time = self.x_train.shape[0] + self.x_val.shape[0]
            # self.y_val = self.data_scaler.scale_y(self.y_val)

        else:
            self.y_val = None
            self.pre_test_data_info = self.training_data_info
            self.starting_test_time = self.x_train.shape[0]

        self.x_test, self.y_test, self.test_data_info = self.data_generator.generate_data(int(self.test_ratio * T),
                                                                                          device=self.device,
                                                                                          previous_data_info=self.pre_test_data_info)
        if len(self.y_test.shape) == 1:
            self.y_test = self.y_test.unsqueeze(-1)

        if self.y_val is not None:
            all_y = torch.cat([self.y_train, self.y_val, self.y_test], dim=0)
        else:
            all_y = torch.cat([self.y_train, self.y_test], dim=0)

        all_y_scaled = self.data_scaler.scale_y(all_y)

        self.y_scaled_min = all_y_scaled.min().item()
        self.y_scaled_max = all_y_scaled.max().item()
        self.y_dim = self.y_train.shape[-1]
        self.x_dim = self.x_train.shape[-1]


def GetDataset(name, base_path):
    """ Load a dataset

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/RCOL_datasets/directory/"

    Returns
    -------
    X : features (nXp)
    y : labels (n)

	"""
    if name == 'energy':
        df = pd.read_csv(base_path + 'energy.csv')
        y = np.array(df['Appliances'])
        X = df.drop(['Appliances', 'date'], axis=1)
        date = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    if name == 'tetuan_power':
        df = pd.read_csv(base_path + 'tetuan_power.csv')
        y = np.array(df['Zone 1 Power Consumption'])
        X = df.drop(['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime'],
                    axis=1)
        date = pd.to_datetime(df['DateTime'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%m/%d/%Y %H:%M')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    if name == 'traffic':
        df = pd.read_csv(base_path + 'traffic.csv')
        df['holiday'].replace(df['holiday'].unique(),
                              list(range(len(df['holiday'].unique()))), inplace=True)
        df['weather_description'].replace(df['weather_description'].unique(),
                                          list(range(len(df['weather_description'].unique()))), inplace=True)
        df['weather_main'].replace(['Clear', 'Haze', 'Mist', 'Fog', 'Clouds', 'Smoke', 'Drizzle', 'Rain', 'Squall',
                                    'Thunderstorm', 'Snow'],
                                   list(range(len(df['weather_main'].unique()))), inplace=True)
        y = np.array(df['traffic_volume'])
        X = df.drop(['date_time', 'traffic_volume'], axis=1)
        date = pd.to_datetime(df['date_time'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        # X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek

        X = np.array(X)

    if name == 'wind':
        df = pd.read_csv(base_path + 'wind_power.csv')
        date = pd.to_datetime(df['dt'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['dt', 'MW'], axis=1)
        y = np.array(df['MW'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['minute'] = date.dt.minute
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]

    if name == 'prices':
        df = pd.read_csv(base_path + 'Prices_2016_2019_extract.csv')
        # 15/01/2016  4:00:00
        date = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['Date', 'Spot', 'hour'], axis=1)
        y = np.array(df['Spot'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]

    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    except Exception as e:
        raise Exception("invalid dataset")

    return X, y


class I2IDataset(DataSet):
    def __init__(self, data_generator, x_shape, device, T, test_ratio, calibration_starting_update_index):
        super().__init__(data_generator, True, device, T, test_ratio, calibration_starting_update_index)
        self.x_shape = x_shape
        self.y_test = torch.zeros(int(test_ratio * T), self.x_shape[1], self.x_shape[2])
        self.test_valid_idx_mask = torch.zeros(int(test_ratio * T), self.x_shape[1], self.x_shape[2]).bool()
        self.test_image_center = torch.zeros(int(test_ratio * T), 2).int()
        self.test_augmentation_info = [None for _ in range(int(test_ratio * T))]
        self.test_scaling_factors = torch.zeros(int(test_ratio * T))
        self.test_meters_factors = torch.zeros(int(test_ratio * T))

        val_size = 1000
        self.y_val = torch.zeros(val_size, self.x_shape[1], self.x_shape[2])
        self.val_valid_idx_mask = torch.zeros(val_size, self.x_shape[1], self.x_shape[2]).bool()
        self.val_image_center = torch.zeros(val_size, 2).int()
        self.val_scaling_factors = torch.zeros(val_size)
        self.val_meters_factors = torch.zeros(val_size)
        self.y_dim = 1
        self.x_dim = self.x_shape[0]
        self.train_size = T - int(test_ratio * T)
        self.test_size = int(test_ratio * T)
        self.val_size = val_size
        self.val_augmentation_info = [None for _ in range(val_size)]

    @abc.abstractmethod
    def get_data(self, index, apply_augmentation):
        pass

    @property
    def x_train(self):
        return None

    @property
    def x_test(self):
        return None

    @property
    def y_train(self):
        return None

    def generate_data(self, T, calibration_starting_update_index):
        train_ratio = 1 - self.test_ratio
        _, _, self.training_data_info = self.data_generator.generate_data(int(train_ratio * T))
        self.y_val = None
        self.pre_test_data_info = self.training_data_info
        self.starting_test_time = int(train_ratio * T)

        _, _, self.test_data_info = self.data_generator.generate_data(int(self.test_ratio * T), device=self.device)


class LazyDepthDataSet(I2IDataset):
    def __init__(self, data_generator: DepthDataGenerator, device, T, test_ratio, calibration_starting_update_index):
        x_shape = data_generator.depth_dataset.get_data(0, apply_augmentation=False)['rgb'].shape
        super().__init__(data_generator, x_shape, device, T, test_ratio, calibration_starting_update_index)

    def get_data(self, index, apply_augmentation):
        data = self.data_generator.depth_dataset.get_data(index, apply_augmentation)
        data['y'] = data['depth']
        starting_val_time = self.train_size - self.val_size
        if not apply_augmentation and starting_val_time <= index < self.train_size:
            self.y_val[index - self.train_size] = data['depth'].squeeze()
            self.val_valid_idx_mask[index - starting_val_time] = data['valid_image_mask'].squeeze().bool()
            self.val_image_center[index - starting_val_time, 0] = data['center_pixel'][0]
            self.val_image_center[index - starting_val_time, 1] = data['center_pixel'][1]
            self.val_scaling_factors[index - starting_val_time] = data['scaling_factor']
            self.val_meters_factors[index - starting_val_time] = data['meters_factor']
            self.val_augmentation_info[index - starting_val_time] = data['augmentation_info']

        if not apply_augmentation and index >= self.train_size:
            self.y_test[index - self.train_size] = data['depth'].squeeze()
            self.test_valid_idx_mask[index - self.train_size] = data['valid_image_mask'].squeeze().bool()
            self.test_image_center[index - self.train_size, 0] = data['center_pixel'][0]
            self.test_image_center[index - self.train_size, 1] = data['center_pixel'][1]
            self.test_scaling_factors[index - self.train_size] = data['scaling_factor']
            self.test_meters_factors[index - self.train_size] = data['meters_factor']
            self.test_augmentation_info[index - self.train_size] = data['augmentation_info']
        return data
