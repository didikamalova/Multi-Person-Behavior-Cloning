import torch.optim as optim
from collections import OrderedDict
import matplotlib.pyplot as plt
from models.encoder import ImagePolicy
import roboverse as rv
import rlkit.torch.pytorch_util as ptu
from moviepy.editor import *
import numpy as np
import collections
import time
import json
import os
import csv
import pdb


def add_fake_person_id(dataset, num_people=5):
	person_id = np.zeros((dataset['observations'].shape[0], num_people))
	num_demos = person_id.shape[0]
	rand_people = np.random.randint(0,5,size = num_demos)
	person_id[np.arange(num_demos), rand_people] = 1
	person_id = np.concatenate([person_id.reshape(person_id.shape[0], 1, person_id.shape[1]) for k in range(dataset['observations'].shape[1])], axis=1)
	dataset['person_id'] = person_id

	return dataset

class InfiniteBatchLoader:
	"""Wraps a Dataset"""
	def __init__(self, dataset, batch_size, normalize=True):
		self.dataset = dataset
		self.batch_size = batch_size
		self.num_traj = dataset['observations'].shape[0]
		self.horizon = dataset['observations'].shape[1]
		self.normalize = normalize

	def __len__(self):
		return len(self.dataset_loader)

	def random_batch(self):
		traj_ind = np.random.choice(self.num_traj, size=(self.batch_size,))
		trans_ind = np.random.choice(self.horizon, size=(self.batch_size,))
		obs = self.dataset['observations'][traj_ind, trans_ind, :]
		acts = self.dataset['actions'][traj_ind, trans_ind, :]
		person_id = self.dataset['person_id'][traj_ind, trans_ind, :]
		if self.normalize: obs = obs / 255.
		return ptu.from_numpy(obs),  ptu.from_numpy(acts),  ptu.from_numpy(person_id)

### USE THIS TO VISUALIZE A DATA POINT ###
# img, act = InfiniteBatchLoader(train_dataset, 1).random_batch()
# img = img.reshape(3, 128, 96).transpose()
# plt.figure()
# plt.imshow(img)
# plt.title("Image Sample")
# plt.show()

class BCTrainer:
	def __init__(self, model=None, train_data=None, test_data=None, log_dir='', env=None, horizon=65, num_epochs=25,
			batch_size=32, weight_decay=0.0, lr=1e-3, num_eval_traj=5, video_logging_period=1, eval_person_id = None):
		self.model = model
		self.num_epochs = num_epochs
		self.eval_person_id = eval_person_id

		params = list(self.model.parameters())
		self.optimizer = optim.Adam(params,lr=lr, weight_decay=weight_decay)
		self.train_dataset = InfiniteBatchLoader(train_data, batch_size=batch_size)
		self.test_dataset = InfiniteBatchLoader(test_data, batch_size=batch_size)

		self.env = env
		self.horizon = horizon
		self.num_eval_traj = num_eval_traj

		self.video_logging_period = video_logging_period
		self.persistent_statistics = collections.defaultdict(list)
		self.eval_statistics = collections.defaultdict(list)
		self.log_dir = log_dir

	def compute_loss(self, X, y, person_id, test=False):
		prefix = "test-" if test else "train-"
		bc_loss = self.model.compute_loss(X, y, person_id)
		self.eval_statistics[prefix + "BC Loss"].append(bc_loss.item())
		return bc_loss

	def evaluate_policy(self, epoch):
		person_id = np.zeros(5)
		person_id[self.eval_person_id] = 1
		person_id = ptu.from_numpy(person_id)

		images = []
		self.env.reset()
		for key in self.env.get_info().keys():
			self.eval_statistics['env-' + key] = []
			self.eval_statistics['env-final-' + key] = []

		for j in range(self.num_eval_traj):
			#print('Finished traj')
			self.env.reset()
			returns = 0
			for i in range(self.horizon):
				rendered_obs = self.env.render_obs()
				images.append(np.uint8(rendered_obs))
				img = ptu.from_numpy(np.uint8(rendered_obs.transpose()).flatten() / 255.)
				action = ptu.get_numpy(self.model(img, person_id)).flatten()
				next_observation, reward, done, info = self.env.step(action)
				for key in info.keys():
					self.eval_statistics['env-' + key].append(info[key])
					if (i + 1) == self.horizon:
						self.eval_statistics['env-final-' + key].append(info[key])
				returns += reward

			self.eval_statistics["Avg Returns"].append(returns)
			self.eval_statistics["Success Rate"].append(info['task_achieved'])

		if epoch % self.video_logging_period == 0:
			video_filename = self.log_dir + 'videos/rollout_{0}.mp4'.format(epoch)
			video = ImageSequenceClip(images, fps=24)
			video.write_videofile(video_filename)

	def train_batch(self, X, y, person_id):
		self.model.train()
		self.optimizer.zero_grad()
		loss = self.compute_loss(X, y, person_id)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
 
	def test_batch(self, X, y, person_id):
		self.model.eval()
		loss = self.compute_loss(X, y, person_id, test=True)

	def train_epoch(self,batches=100):
		start_time = time.time()
		for b in range(batches):
			X, y, person_id = self.train_dataset.random_batch()
			self.train_batch(X, y, person_id)
		self.eval_statistics["train-epoch_duration"].append(time.time() - start_time)

	def test_epoch(self,batches=10):
		start_time = time.time()
		for b in range(batches):
			X, y, person_id= self.test_dataset.random_batch()
			#import pdb; pdb.set_trace()
			self.test_batch(X, y, person_id)
		self.eval_statistics["test-epoch_duration"].append(time.time() - start_time)

	def output_diagnostics(self,epoch):
		stats = OrderedDict()
		for k in sorted(self.eval_statistics.keys()):
			stats[k] = np.mean(self.eval_statistics[k])
			self.persistent_statistics[k + '/mean'].append(stats[k])
			self.persistent_statistics[k + '/std'].append(np.std(self.eval_statistics[k]))

		self.update_plots(epoch)

		if epoch == 0:
			with open(self.log_dir + 'progress.csv', 'w', newline='') as f:
				writer = csv.DictWriter(f, fieldnames=stats.keys())
				writer.writeheader()

		with open(self.log_dir + 'progress.csv', 'a', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=stats.keys())
			writer.writerow(stats)

		self.eval_statistics = collections.defaultdict(list)
		np.save(self.log_dir + 'logs.npy', self.persistent_statistics)

		print("\nEPOCH: ", epoch)
		for k, v in stats.items():
			spacing = ":" + ' ' * (30 - len(k))
			print(k + spacing + str(round(v, 3)))

	def update_plots(self, epoch, num_avg=3):
		x_axis = np.arange(epoch + 1)
		print(x_axis)
		for k in sorted(self.eval_statistics.keys()):
			plt.clf()
			mean = np.array(self.persistent_statistics[k + '/mean'])
			std = np.array(self.persistent_statistics[k + '/std'])
			plt.plot(x_axis, mean, color='blue')
			plt.fill_between(x_axis, mean - std, mean + std, facecolor='blue', alpha=0.5)
			plt.title(k)
			plt.savefig(self.log_dir + 'graphs/{0}.png'.format(k))

	def train(self):
		for epoch in range(self.num_epochs):
			self.train_epoch()
			#print('Finished training')
			self.test_epoch()
			#print('Finished testing')
			self.evaluate_policy(epoch)
			#print('Finished evaluating')
			self.output_diagnostics(epoch)

def make_datasets(filepath, num_traj_limit=300, train_percent=0.9, use_noisy_actions = True):
	dataset = np.load(filepath, allow_pickle=True).item()
	# dataset = add_fake_person_id(dataset)

	num_samples = dataset['observations'].shape[0]
	train_end = int(num_samples * train_percent)

	ind = np.random.choice(num_samples, size=(num_samples,), replace=False)
	train_ind, test_ind = ind[:train_end], ind[train_end:]
	action_key = 'noisy_actions'if use_noisy_actions else 'actions'
	train_dataset = {'observations': dataset['observations'][train_ind],
					'actions': dataset[action_key][train_ind],
					'person_id': dataset['person_id'][train_ind],}
	test_dataset = {'observations': dataset['observations'][test_ind],
					'actions': dataset[action_key][test_ind],
					'person_id': dataset['person_id'][test_ind],}

	return train_dataset, test_dataset, dataset
def bc_exp_launcher(variant, run_id, exp_id):
	env_name = variant['env_name']
	horizon = variant['horizon']
	datapath = variant['datapath']
	log_dir = variant['log_dir'] + 'run{0}/id{1}/'.format(run_id, exp_id)
	folder_exists = os.path.exists(log_dir)
	
	if not folder_exists:
		os.makedirs(log_dir)
		os.makedirs(log_dir + 'videos/')
		os.makedirs(log_dir + 'graphs/')
	
	with open(log_dir + "variant.json", "w") as outfile:
		json.dump(variant, outfile)

	ptu.set_gpu_mode(variant['use_gpu'])
	model = ImagePolicy(**variant['model_kwargs'])
	env = rv.make(env_name, **variant['env_kwargs'])
	train_dataset, test_dataset, dataset = make_datasets(datapath,num_traj_limit=variant['demo_size'])
	trainer = BCTrainer(model=model, train_data=train_dataset, test_data=test_dataset,
		log_dir=log_dir, env=env, horizon=horizon, eval_person_id=variant['eval_person_id'])
	trainer.train()