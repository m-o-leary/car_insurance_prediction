import argparse
from rich.console import Console
from trainer_lib import DataManager
from trainer_lib.utils.notebook_config import DATA_DIR, REPORT_DIR, DOCKER_DATA_DIR, MODEL_DIR, REPORT_DIR
from trainer_lib.modelling.train import Trainer, Evaluation, Explain
from trainer_lib.utils.filesystem import is_dir, make_dir
from trainer_lib.modelling.model_config import GRID_SEARCH, MODEL_POST_URL, PRE_PROCESSING_STEPS
import os 
from datetime import datetime
import warnings

# Deprecations and pandas slice warnings due to the train_test split
warnings.filterwarnings('ignore')

def get_parser():
	"""
	Creates a new argument parser.
	"""
	parser = argparse.ArgumentParser('trainer_lib')
	
	parser.add_argument('experiment',
		action="store",\
		help="Name of the experiment")
	parser.add_argument('--profile',
		action="store_true",
		help=f"Profile the data and save to {REPORT_DIR} directory.")
	parser.add_argument('--out',
		action="store",
		help="If provided, will store the transformed data in this location.")
	
	model_list = list(GRID_SEARCH.keys())
	model_list_str = ', '.join(model_list)
	parser.add_argument('--model',
		nargs="*",
		default=model_list,
		action="store",
		choices=model_list,
		help=f"""Specify a list of models to train.
		By default will train the following list:
		{model_list_str}.
		""")

	return parser

def main(args=None):
	"""
	Main entry point for training.
	"""

	parser = get_parser()
	args = parser.parse_args(args)
	console = Console()
	# console.print(args)
	
	mngr = DataManager(save_path=DOCKER_DATA_DIR, report_path=REPORT_DIR )
	X, y = mngr.train
	X_test, _ = mngr.test

	# Train pipeline  
	start = datetime.now()
	search = { k:v for k,v in GRID_SEARCH.items() if k in args.model }
	
	trainer = Trainer(X, y.values.ravel(), 
					  grid=search, 
					  save_path=MODEL_DIR, 
					  pre_processing=PRE_PROCESSING_STEPS)

	trainer.train(args.experiment)
	time = datetime.now() - start
	console.print(f"Training [green]done[/green] in {time}.")

	if args.profile:
		# Profile the datasets and save to ../reports/
		
		start = datetime.now()
		with console.status("Profiling datasets."):
			mngr.profile()
		time = datetime.now() - start
		console.print(f"Profiling [green]done[/green] in {time}.")
		console.print("fReports available in ")

	if args.out:
		# Check if dir exists
		if is_dir(args.out) is False:
			make_dir(args.out)	
		# Save the processed dataset to the provided location.
		score, params, hash_id = trainer.performance[trainer.best_classifier__name]
		start = datetime.now()
		with console.status(f"Saving processed dataset to {args.out}."):
			saved_file = f"data_{hash_id}.pkl"
			(trainer.
				best_classifier
				.named_steps['pre_processing']
				.named_steps['feature_engineering']
				.transform(X)
				.to_csv(os.path.join(args.out, saved_file)))
		time = datetime.now() - start
		console.print(f"Profiling [green]done[/green] in {time}.")
		console.print(f"Processed dataset saved to {args.out}/{saved_file} ")

if __name__ == '__main__':
    main()
