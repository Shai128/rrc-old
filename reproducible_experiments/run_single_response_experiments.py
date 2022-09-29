import subprocess

SEEDS = 20


def main():
    for i in range(SEEDS):
        python_cmd = f'python main.py --cal_split=0 --seed={i}'
        print(f'Running: {python_cmd}')
        subprocess.check_call(python_cmd, shell=True)

        for dataset in ['tetuan_power', 'energy', 'traffic', 'wind', 'prices']:
            python_cmd = f'python main.py --ds_type=REAL --dataset_name={dataset} --cal_split=0 --seed={i}'
            print(f'Running: {python_cmd}')
            subprocess.check_call(python_cmd, shell=True)


if __name__ == '__main__':
    main()
