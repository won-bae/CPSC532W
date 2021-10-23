import os
import numpy as np
import matplotlib.pyplot as plt

eval_dir = 'results/evaluation'
graph_dir = 'results/graph'

eval_mean = np.load(os.path.join(graph_dir, 'means.npy'), allow_pickle=True)


shapes = [[1, 1], [1, 2], [2, 10], [1, 10], [1, 10], [10, 10], [1, 10]]
titles = ['mu',
			['slope', 'bias'],
			['state1', 'state2', 'state3', 'state4', 'state5',
			 'state6', 'state7', 'state8', 'state9', 'state10',
			 'state11', 'state12', 'state13', 'state14', 'state15',
			 'state16', 'state17'],
			['W0'],
			['b0'],
			['W1'],
			['b1']]

file_names = ['samples1.npy', 'samples2.npy', 'samples3.npy', 'samples4.npy',]
figsize = 5


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


for i in range(len(shapes)):
	shape = shapes[i]
	title = titles[i]
	if i >= 3:
		file_name = file_names[3]
	else:
		file_name = file_names[i]

	sample = np.load(os.path.join(graph_dir, file_name), allow_pickle=True)
	if i == 3:
		sample = sample[:,:,0]
	elif i == 4:
		sample = sample[:,:,1]
	elif i == 5:
		sample = sample[:,:,2:12]
	elif i == 6:
		sample = sample[:,:,12]

	# import IPython; IPython.embed()
	# axs = trim_axs(axs, len(cases))

	# if i == 0:
	# 	axes = plt.figure(figsize=(figsize, figsize), constrained_layout=True).subplots(shape[0], shape[1])
	# 	axes.set_title(title)
	# 	axes.hist(sample)
	# if i == 1:
	# 	axes = plt.figure(figsize=(figsize*2, figsize), constrained_layout=True).subplots(shape[0], shape[1])
	# 	for j, ax in enumerate(axes):
	# 		sample_i = sample[:,j]
	# 		title_i = title[j]
	# 		ax.set_title(title_i)
	# 		ax.hist(sample_i)
	# if i == 2:
	# 	axes = plt.figure(figsize=(figsize*5, figsize), constrained_layout=True).subplots(shape[0], shape[1])
	# 	for j in range(20):
	# 		if j >= 17: break
	# 		row = j // shape[1]
	# 		col = j % shape[1]
	# 		sample_i = sample[:,j]
	# 		# import IPython; IPython.embed(); exit()
	# 		title_i = title[j]
	# 		axes[row, col].set_title(title_i)
	# 		axes[row, col].hist(sample_i)
	# if i == 3:
	# 	axes = plt.figure(figsize=(figsize*10, figsize), constrained_layout=True).subplots(shape[0], shape[1])
	# 	for j in range(10):
	# 		sample_i = sample[:,j]
	# 		axes[j].set_title(title[0] + '_' + str(j))
	# 		axes[j].hist(sample_i)
	# if i == 4:
	# 	axes = plt.figure(figsize=(figsize*10, figsize), constrained_layout=True).subplots(shape[0], shape[1])
	# 	for j in range(10):
	# 		sample_i = sample[:,j]
	# 		axes[j].set_title(title[0] + '_' + str(j))
	# 		axes[j].hist(sample_i)
	if i == 5:
		axes = plt.figure(figsize=(figsize*10, figsize* 2), constrained_layout=True).subplots(shape[0], shape[1])
		for j in range(10):
			for k in range(10):
				sample_i = sample[:,j, k]
				if j == 0:
					axes[j, k].set_title(title[0] + '_' + str(k))
				axes[j, k].hist(sample_i)
	# if i == 6:
	# 	axes = plt.figure(figsize=(figsize*10, figsize), constrained_layout=True).subplots(shape[0], shape[1])
	# 	for j in range(10):
	# 		sample_i = sample[:,j]
	# 		axes[j].set_title(title[0] + '_' + str(j))
	# 		axes[j].hist(sample_i)
	else:
		continue
		# axes = plt.figure(figsize=(figsize, figsize), constrained_layout=True).subplots(shape[0], shape[1])


	plt.show()