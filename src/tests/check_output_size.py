import os
import argparse
import glob 
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, help='model name to be used for saving the model', default='mvo_ls_small_sample_mle_mle')


if __name__ == '__main__':
    args = parser.parse_args()
    threshold = 100

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'outputs', args.model_name, "*.pickle")
    pkl_files = glob.glob(path)

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'outputs', args.model_name, "*.csv")
    csv_files = glob.glob(path)

    files = pkl_files + csv_files

    for f in tqdm(files, total=len(files), desc="Checking output size of {} reports".format(args.model_name)):
        if (os.path.getsize(f) / 1e6) > threshold:
            print(f)
