from dachi.behavior import _build as build
from dachi.behavior import _tasks as tasks, Not, Sequence

from .test_behavior import ATask


class TestParallel:

    def test_parallel_creates_a_parallel_node(self):

        with build.parallel() as parallel:

            parallel.add(ATask())
            parallel.add(ATask())
        assert isinstance(parallel, tasks.Parallel)

    def test_parallel_creates_a_parallel_node_with_correct_fails_on(self):

        with build.parallel() as parallel:

            parallel.add(ATask())
            parallel.add(ATask())
            parallel.fails_on = 2
        assert parallel.fails_on == 2


class TestSelector:

    def test_selector_creates_a_parallel_node(self):

        with build.select() as selector:

            selector.add(ATask())
            selector.add(ATask())
        assert isinstance(selector, tasks.Selector)


class TestSequence:

    def test_sequence_creates_a_parallel_node(self):

        with build.sequence() as sequence:

            sequence.add(ATask())
            sequence.add(ATask())
        assert isinstance(sequence, tasks.Sequence)


class TestSango:

    def test_sango_creates_a_tree(self):

        with build.sango() as tree:

            with build.sequence(tree) as sequence:

                sequence.add(ATask())
                sequence.add(ATask())
        assert isinstance(tree, tasks.Sango)
        assert isinstance(tree.root, Sequence)


class TestWhile:

    def test_sango_creates_a_tree(self):


        with build.while_(build.sequence()) as while_:

            while_.task.add(ATask())
            while_.task.add(ATask())
        assert isinstance(while_, tasks.While)


class TestUntil:

    def test_sango_creates_a_tree(self):

        with build.until_(build.sequence()) as until_:

            until_.task.add(ATask())
            until_.task.add(ATask())
        assert isinstance(until_, tasks.Until)


class TestNot:

    def test_not_creates_a_negative_node(self):

        with build.not_(build.sequence()) as not_:

            not_.task.add(ATask())
            not_.task.add(ATask())
        assert isinstance(not_, tasks.Not)

    def test_not_decorates_a_sequence(self):

        with build.not_(build.sequence()) as not_:

            not_.task.add(ATask())
            not_.task.add(ATask())
        assert isinstance(not_.task, tasks.Sequence)
