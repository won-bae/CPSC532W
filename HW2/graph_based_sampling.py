import torch
import torch.distributions as dist

from daphne import daphne

# from primitives import funcPrimitives #TODO
from tests import is_tol, run_prob_test,load_truth
from primitives import Primitives


# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
# env = {'normal': dist.Normal,
#        'sqrt': torch.sqrt}

primitives = Primitives()
env = primitives.get_func_maps()

def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


def _deterministic_eval(exp, graph):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list and exp[0] == 'if':
        cond = _deterministic_eval(exp[1], graph)
        if cond:
            return _deterministic_eval(exp[2], graph)
        return _deterministic_eval(exp[3], graph)
    elif type(exp) is list:
        op = exp[0]
        args = exp[1:]
        # try:
        result = env[op](*map(_deterministic_eval, args, [graph] * len(args)))
        # except:
            # import IPython; IPython.embed()
        return result
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    elif exp in graph[1]['V']:
        exp = graph[1]['P'][exp]
        return _deterministic_eval(exp, graph)
    else:
        raise("Expression type unknown.", exp)



def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    E = graph[-1]
    results = _deterministic_eval(E, graph)
    return results


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../programs/tests/deterministic/test_{}.daphne'.format(i)])
        print("{}th test ast: {}".format(i, graph))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../programs/tests/probabilistic/test_{}.daphne'.format(i)])
        print("{}th test ast: {}".format(i, graph))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    # run_deterministic_tests()
    # run_probabilistic_tests()

    num_samples = 1000

    for i in range(1,5):
        results = []
        for _ in range(num_samples): 
            graph = daphne(['graph','-i','../programs/{}.daphne'.format(i)])
            # print('\n\n\nSample of prior of program {}:'.format(i))
            # print("{}th test ast: {}".format(i, graph))
            result = sample_from_joint(graph)  
            results.append(result)

        if (i+1) % 100 == 0:
            print('{}/{}'.format(i+1, num_samples))

        import IPython; IPython.embed()
        results = torch.stack(results)
        print(results.shape)

    