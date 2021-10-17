import torch
import torch.distributions as dist
#TODO


class Primitives(object):

	def __init__(self):
		self._primitive_func = {
			"+": lambda *x: torch.sum(torch.tensor(x)),
			"sqrt": lambda x: torch.sqrt(torch.tensor(x)),
			"*": lambda *x: torch.prod(torch.tensor(x)),
			"/": div,
			"<": less,
			"vector": vector,
			"get": get,
			"put": put,
			"first": first,
			"last": last,
			"rest": rest,
			"append": append,
			"hash-map": hash_map,
			"normal": normal,
			"beta": beta,
			"exponential": exponential,
			"uniform": uniform,
			"discrete": discrete,
			"sample*": lambda x: x.sample().item(),
			"observe*": lambda x: [],
			"mat-transpose": mat_transpose,
			"mat-mul": mat_mul,
			"mat-tanh": mat_tanh,
			"mat-add": mat_add,
			"mat-repmat": mat_repmat,

		}
	def is_primitive(self, string):
		if string in self._primitive_func:
			return True
		return False

	def to_func(self, string):
		return self._primitive_func[string]

	def get_func_maps(self):
		return self._primitive_func


# functions
def less(*x):
	return x[0] < x[1]

def div(*x):
	return torch.div(x[0], x[1])

def vector(*x):
	if isinstance(x[0], list) and len(x[0]) == 0:
		return torch.tensor([])

	vectorized = []
	dim = 0
	prev = None
	for i, xi in enumerate(x):
		if isinstance(xi, (float, int)):
			vectorized.append(torch.tensor([xi]))
		elif isinstance(xi, torch.Tensor) and len(xi.shape) <= 1:
			vectorized.append(xi.unsqueeze(0))
		elif isinstance(xi, torch.Tensor):
			if i == 0:
				prev, dim = xi, 1
			else:
				dim = dim and (prev.shape[0] == xi.shape[0])
			vectorized.append(xi)
		else:
			vectorized.append(xi)			
	try:
		vectorized = torch.cat(vectorized, dim=int(dim)).float()
	except:
		vectorized = vectorized
	return vectorized

def get(*x):
	lst_or_dict, idx = x[0], int(x[1])
	return lst_or_dict[idx]

def put(*x):
	lst_or_dict, idx, val = x[0], x[1], torch.tensor(x[2])
	lst_or_dict[idx] = val
	return lst_or_dict

def first(*x):
	return x[0][0]

def last(*x):
	return x[0][-1]

def rest(*x):
	return x[0][1:]

def append(*x):
	lst, val = x[0], torch.tensor([x[1]])
	return torch.cat((lst, val))

def hash_map(*x):
	# import IPython; IPython.embed()
	hashmap = {}
	for i in range(0, len(x), 2):
		hashmap[x[i]] = torch.tensor(x[i+1])
	return hashmap

def normal(*x):
	mu, sigma = x[0], x[1]
	sampler = dist.normal.Normal(mu, sigma)
	return sampler
	 
def beta(*x):
	alpha, gamma = x[0], x[1]
	sampler = dist.beta.Beta(alpha, gamma)
	return sampler

def exponential(*x):
	rate = x[0]
	sampler = dist.exponential.Exponential(rate)
	return sampler

def uniform(*x):
	low, high = x[0], x[1]
	sampler = dist.uniform.Uniform(low, high)
	return sampler

def discrete(*x):
	sampler = dist.categorical.Categorical(x[0])
	return sampler

def mat_transpose(*x):
	return torch.transpose(x[0], 1, 0)

def mat_mul(*x):
	return torch.matmul(x[0], x[1])

def mat_tanh(*x):
	return torch.tanh(x[0])

def mat_add(*x):
	return x[0] + x[1]

def mat_repmat(*x):
	return torch.tile(x[0], (x[1], x[2]))

