import argparse
from rich.console import Console
from . import KaggleCarInsuranceDataLoader
from . import DataProfiler

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser('Car Insurance Model Trainer Lib')
    parser.add_argument('--fetch_data',
						nargs="?",
						default=False,
                        help='Fetch the data from remote URL.')
    parser.add_argument('--train', 
						nargs="?",
						default=False,
						help="Train the estimator")
    parser.add_argument('--profile', 
						nargs="?",
						default=False,
						help="Profile the data.")
    parser.add_argument('--model',
						nargs="?",
						default="logistic_regression",
						choices=['logistic_regression', 'random_forest', 'xgboost'],
						help="Select a model to train. (default: logistic_regression).")
    return parser

def main(args=None):
	"""
	Main entry point for pipeline.
	Args:
	args : list
			A of arguments as if they were input in the command line. Leave it
			None to use sys.argv.
	"""

	parser = get_parser()
	args = parser.parse_args(args)
	console = Console()

	kaggle = KaggleCarInsuranceDataLoader("./data/raw", fetch=args.fetch_data)
	
	if args.train:
		console.print("Training model...")
		pass
	if args.profile:
		console.print("Profiling datasets...")
		DataProfiler(data=kaggle.get_train_data(return_x_y=False), 
					 report_title='train_data', 
					 out_path="./reports").profile()
		DataProfiler(data=kaggle.get_test_data(return_x_y=False), 
					 report_title='train_data', 
					 out_path="./reports").profile()

if __name__ == '__main__':
    main()
