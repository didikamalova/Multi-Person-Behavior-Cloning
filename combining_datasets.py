import numpy as np

filename_function = lambda id: '/Users/macintosh/Desktop/Experiment/datasets/tray_demos_person_{0}.npy'.format(id)
all_files = [filename_function(i) for i in range(5)]
datasets = [np.load(f, allow_pickle=True).item() for f in all_files]

combined_dataset = {}

def merge_datasets(combined_dataset, datasets):
	for dataset in datasets:
		for key in dataset.keys():
			if key in combined_dataset:
				combined_dataset[key] = np.concatenate([combined_dataset[key], dataset[key]], axis=0)
			else:
				combined_dataset[key] = dataset[key].copy()

merge_datasets(combined_dataset, datasets)

np.save('/Users/macintosh/Desktop/Experiment/datasets/conditioned_combined_dataset', combined_dataset)

combined_dataset['person_id'] *= 0

np.save('/Users/macintosh/Desktop/Experiment/datasets/unconditioned_combined_dataset', combined_dataset)