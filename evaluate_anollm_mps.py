import os
from pathlib import Path
import argparse

import numpy as np
import torch
import time

#import torch.distributed as dist

from anollm import AnoLLM
from src.data_utils import load_data, DATA_MAP, get_text_columns, get_max_length_dict
from train_anollm_mps import get_run_name


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type = str, default='wine', choices = [d.lower() for d in DATA_MAP.keys()],
					help="Name of datasets in the ODDS benchmark")
	parser.add_argument("--exp_dir", type = str, default=None)
	parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'], help="semi_supervised:an uncontaminated, unsupervised setting; unsupervised:a contaminated, unsupervised setting")
	
	#dataset hyperparameters
	parser.add_argument("--data_dir", type = str, default='data')
	parser.add_argument("--n_splits", type = int, default=5)
	parser.add_argument("--split_idx", type = int, default=None) # 0 to n_split-1
	# binning
	parser.add_argument("--binning", type = str, choices=['quantile', 'equal_width', 'language', 'none', 'standard'], default='standard')
	parser.add_argument("--n_buckets", type = int, default=10)
	parser.add_argument("--remove_feature_name", action = 'store_true')
	
	# model hyperparameters (for getting the model name)
	parser.add_argument("--model", type = str, choices = ['gpt2', 'distilgpt2', 'smol', 'smol-360', 'smol-1.7b'], default='smol')
	parser.add_argument("--lora", action='store_true', default=False)
	parser.add_argument("--lr", type = float, default=5e-5)
	parser.add_argument("--random_init", action='store_true', default=False)
	parser.add_argument("--no_random_permutation", action='store_true', default=False)
	
	#testing
	parser.add_argument("--batch_size", type = int, default=128) # per gpu
	parser.add_argument("--n_permutations", type = int, default=100) # per gpu
	args = parser.parse_args()
	
	if args.model == 'smol':
		args.model = 'HuggingFaceTB/SmolLM-135M'
	elif args.model == 'smol-360':
		args.model = 'HuggingFaceTB/SmolLM-360M'
	elif args.model == 'smol-1.7b':	
		args.model = 'HuggingFaceTB/SmolLM-1.7B'
	
	return args

def main():
	args = get_args()

	if torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	print("Using device:", device)

	if args.exp_dir is None:
		args.exp_dir = (
				Path("exp")
				/ args.dataset
				/ args.setting
				/ f"split{args.n_splits}"
				/ f"split{args.split_idx}"
		)

	if not os.path.exists(args.exp_dir):
		raise ValueError(
			f"Experiment directory {args.exp_dir} does not exist"
		)

	score_dir = args.exp_dir / "scores"
	os.makedirs(score_dir, exist_ok=True)

	run_name = get_run_name(args)
	score_path = score_dir / f"{run_name}.npy"

	print("score_path:", score_path)

	X_train, X_test, y_train, y_test = load_data(args)

	if os.path.exists(score_path):
		print("Scores already exist, skip inference")
		return

	model_dir = args.exp_dir / "models"
	model_path = model_dir / f"{run_name}.pt"

	if not os.path.exists(model_path):
		raise FileNotFoundError(
			f"Trained model not found: {model_path}"
		)

	efficient_finetuning = "lora" if args.lora else ""
	max_length_dict = get_max_length_dict(args.dataset)
	text_columns = get_text_columns(args.dataset)

	model = AnoLLM(
		args.model,
		efficient_finetuning=efficient_finetuning,
		model_path=model_path,
		max_length_dict=max_length_dict,
		textual_columns=text_columns,
		no_random_permutation=args.no_random_permutation,
		bf16=False,
	)

	print("Text columns:", text_columns)
	print("Maximum lengths:", max_length_dict)

	model.load_from_state_dict(model_path)
	model.model.to(device)
	model.model.eval()

	start_time = time.time()

	scores = model.decision_function(
		X_test,
		n_permutations=args.n_permutations,
		batch_size=args.batch_size,
		device=str(device),
	)

	end_time = time.time()

	print("Inference time:", end_time - start_time)
	print("Raw score shape:", scores.shape)

	run_time_dir = args.exp_dir / "run_time" / "test"
	os.makedirs(run_time_dir, exist_ok=True)

	run_time_path = run_time_dir / f"{run_name}.txt"
	with open(run_time_path, "w") as file:
		file.write(str(end_time - start_time))

	# decision_function 返回：
	# [测试样本数量, permutation 数量]
	mean_scores = np.mean(scores, axis=1)

	np.save(score_path, mean_scores)

	raw_score_path = score_dir / f"raw_{run_name}.npy"
	np.save(raw_score_path, scores)

	print("Saved mean scores to:", score_path)
	print("Saved raw scores to:", raw_score_path)


if __name__ == "__main__":
	main()
