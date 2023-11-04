import argparse

from data.ETFsLoader import ETFsLoader

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', type=str, help='To run in debug mode.', default="False")

if __name__ == "__main__":
    args = parser.parse_args()

    loader = ETFsLoader()