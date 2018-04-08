import random
import math
import copy
import heapq

from utils import(
    removeall, unique, product, mode, argmax, argmax_random_tie, isclose, gaussian,
    dotproduct, vector_add, scalar_vector_product, weighted_sample_with_replacement,
    weighted_sampler, num_or_str, normalize, clip, sigmoid, print_table,
    open_data, sigmoid_derivative, probability, norm, matrix_multiplication
)

# ______________________________________________________________________________


class DecisionFork:
    """A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values."""

    def __init__(self, attr, attrname=None, default_child=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attrname = attrname or attr
        self.default_child = default_child
        self.branches = branches or {}

    def __call__(self, example):
        """Given an example, classify it using the attribute and the branches."""
        attrvalue = example[self.attr]
        if attrvalue in self.branches:
            return self.branches[attrvalue](example)
        else:
            # return default class when attribute is unknown
            return self.default_child(example)

    def add(self, val, subtree):
        """Add a branch.  If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attrname
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)
        print()   # newline

    def __repr__(self):
        return ('DecisionFork({0!r}, {1!r}, {2!r})'
                .format(self.attr, self.attrname, self.branches))


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display(self, indent=0):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)

# ______________________________________________________________________________


def DecisionTreeLearner(dataset):
    target, values = dataset.target, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        elif all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return plurality_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = DecisionFork(A, dataset.attrnames[A], plurality_value(examples))
            for (v_k, exs) in split_by(A, examples):
                subtree = decision_tree_learning(
                    exs, removeall(A, attrs), examples)
                tree.add(v_k, subtree)
            return tree

    def plurality_value(examples):
        popular = argmax_random_tie(values[target],
                                    key=lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        return sum(e[attr] == val for e in examples)

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs,
                                 key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""
        def I(examples):
            return information_content([count(target, v, examples)
                                        for v in values[target]])
        N = len(examples)
        remainder = sum((len(examples_i)/N) * I(examples_i)
                        for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v])
                for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)


def information_content(values):
    """Number of bits to represent the probability distribution in values."""
    probabilities = normalize(removeall(0, values))
    return sum(-p * math.log2(p) for p in probabilities)