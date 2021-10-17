import torch
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import Primitives
from collections.abc import Hashable
import json


primitives = Primitives()

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    ro = {}
    if 'defn' in ast[0][0]:
        ro[ast[0][1]] = (ast[0][2], ast[0][3])
        result = _evaluate_program(ast[1], {}, {}, ro)
    else:
        result = _evaluate_program(ast[0], {}, {}, ro)
    return result


def _evaluate_program(ast, sigma, local_map, ro):
    if isinstance(ast, (int, float)):
        return ast, sigma
    elif isinstance(ast, Hashable) and ast in local_map:
        return local_map[ast], sigma
    elif ast[0] == 'sample':
        dist, _ = _evaluate_program(ast[1], sigma, local_map, ro)
        sample = dist.sample().item()
        return sample
    elif ast[0] == 'observe':
        return []
    elif ast[0] == 'let':
        assignment = ast[1]
        result = _evaluate_program(assignment[1], sigma, local_map, ro)
        if isinstance(result, tuple):
            val, sigma = result
        else:
            val = result
        local_map[assignment[0]] = val
        return _evaluate_program(ast[2], sigma, local_map, ro)
    elif ast[0] == 'if':
        cond, sigma = _evaluate_program(ast[1], sigma, local_map, ro)
        if cond:
            return _evaluate_program(ast[2], sigma, local_map, ro)
        return _evaluate_program(ast[3], sigma, local_map, ro)
    else:
        vals = []
        for i in range(1, len(ast)):
            result = _evaluate_program(ast[i], sigma, local_map, ro)
            if isinstance(result, tuple):
                val, sigma = result
            else:
                val = result
            vals.append(val)
        if ast[0] in ro:
            vars, e0 = ro[ast[0]]
            for var, val in zip(vars, vals):
                local_map[var] = val
            return _evaluate_program(e0, sigma, local_map, ro)
        if primitives.is_primitive(ast[0]):
            func = primitives.to_func(ast[0])
            # try:
            result = func(*vals)
            # except:
            #     import IPython; IPython.embed()
            return result, sigma


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)



def run_deterministic_tests():
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        #ast = daphne(['desugar', '-i', '../programs/tests/deterministic/test_{}.daphne'.format(i)])
        ast = json.load(open('json/deterministic/test_{}.json'.format(i), 'r'))
        print("{}th test ast: {}".format(i, ast))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))

        print('Test passed')

    print('All deterministic tests passed')



def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-4

    for i in range(1,7):
        #note: this path should be with respect to the daphne path!
        #ast = daphne(['desugar', '-i', '../programs/tests/probabilistic/test_{}.daphne'.format(i)])
        ast = json.load(open('json/probabilistic/test_{}.json'.format(i), 'r'))
        print("{}th test ast: {}".format(i, ast))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')


if __name__ == '__main__':

    run_deterministic_tests()

    run_probabilistic_tests()


    for i in range(1,5):
        #ast = daphne(['desugar', '-i', '../programs/{}.daphne'.format(i)])
        ast = json.load(open('json/{}.json'.format(i), 'r'))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print("{}th test ast: {}".format(i, ast))
        result = evaluate_program(ast)
        if isinstance(result, (int, float)):
            print(result)
        else:
            print(result[0])
