from dachi.behavior import _build as build
from dachi.behavior import _tasks as tasks

from .test_behavior import ATask


class TestParallel:

    def test_parallel_creates_a_parallel_node(self):

        with build.parallel(name='Action') as parallel:

            parallel.add(ATask())
            parallel.add(ATask())
        assert isinstance(parallel.build(), tasks.Parallel)

    def test_parallel_creates_a_parallel_node_with_correct_fails_on(self):

        with build.parallel(name='Action') as parallel:

            parallel.add(ATask())
            parallel.add(ATask())
            parallel.fails_on = 2
        assert parallel.build().fails_on == 2


class TestSelector:

    def test_parallel_creates_a_parallel_node(self):

        with build.select(name='Action') as selector:

            selector.add(ATask())
            selector.add(ATask())
        assert isinstance(selector.build(), tasks.Selector)


class TestSequence:

    def test_parallel_creates_a_parallel_node(self):

        with build.sequence(name='Action') as sequence:

            sequence.add(ATask())
            sequence.add(ATask())
        assert isinstance(sequence.build(), tasks.Sequence)


class TestSango:

    def test_sango_creates_a_tree(self):

        with build.sango('Sango') as tree:

            with build.sequence(tree, name='Action') as sequence:

                sequence.add(ATask())
                sequence.add(ATask())
        assert isinstance(tree.build(), tasks.Sango)

    def test_sango_creates_a_tree_with_a_sequence(self):

        with build.sango(name='Sango') as tree:

            with build.sequence(tree, name='Action') as sequence:

                sequence.add(ATask())
                sequence.add(ATask())
        assert isinstance(tree.build().root, tasks.Sequence)


class TestWhile:

    def test_sango_creates_a_tree(self):


        with build.while_(build.sequence()) as while_:

            while_.add(ATask())
            while_.add(ATask())
        assert isinstance(while_.build(), tasks.While)


class TestUntil:

    def test_sango_creates_a_tree(self):

        with build.until_(build.sequence()) as until_:

            until_.add(ATask())
            until_.add(ATask())
        assert isinstance(until_.build(), tasks.Until)


class TestNot:

    def test_not_creates_a_negative_node(self):

        with build.not_(build.sequence()) as not_:

            not_.add(ATask())
            not_.add(ATask())
        assert isinstance(not_.build(), tasks.Not)

    def test_not_decorates_a_sequence(self):

        with build.not_(build.sequence()) as not_:

            not_.add(ATask())
            not_.add(ATask())
        assert isinstance(not_.build().task, tasks.Sequence)
