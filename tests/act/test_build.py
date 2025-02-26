from dachi.act import _build as build
from dachi.act import _tasks as tasks, Not, Sequence

from .test_tasks import ATask


class TestParallel:

    def test_parallel_creates_a_parallel_node(self):

        with build.build_parallel() as parallel:

            parallel.tasks.append(ATask())
            parallel.tasks.append(ATask())
        assert isinstance(parallel, tasks.Parallel)

    def test_parallel_creates_a_parallel_node_with_correct_fails_on(self):

        with build.build_parallel() as parallel:

            parallel.tasks.append(ATask())
            parallel.tasks.append(ATask())
            parallel.fails_on = 2
        assert parallel.fails_on == 2


class TestSelector:

    def test_selector_creates_a_selector_node(self):

        with build.build_select() as selector:
            
            print(selector)
            selector.tasks.append(ATask())
            selector.tasks.append(ATask())
        assert isinstance(selector, tasks.Selector)


class TestSequence:

    def test_sequence_creates_a_parallel_node(self):

        with build.build_sequence() as sequence:

            sequence.tasks.append(ATask())
            sequence.tasks.append(ATask())
        assert isinstance(sequence, tasks.Sequence)


class TestSango:

    def test_sango_creates_a_tree(self):

        with build.build_sango() as tree:

            with build.build_sequence(tree) as sequence:

                sequence.tasks.append(ATask())
                sequence.tasks.append(ATask())
        assert isinstance(tree, tasks.Root)
        assert isinstance(tree.root, Sequence)


class TestWhile:

    def test_sango_creates_a_tree(self):


        with build.build_unless(build.build_sequence()) as while_:

            while_.task.tasks.append(ATask())
            while_.task.tasks.append(ATask())
        assert isinstance(while_, tasks.Unless)


class TestUntil:

    def test_sango_creates_a_tree(self):

        with build.build_until(build.build_sequence()) as until_:

            until_.task.tasks.append(ATask())
            until_.task.tasks.append(ATask())
        assert isinstance(until_, tasks.Until)


class TestNot:

    def test_not_creates_a_negative_node(self):

        with build.build_not(build.build_sequence()) as not_:

            not_.task.tasks.append(ATask())
            not_.task.tasks.append(ATask())
        assert isinstance(not_, tasks.Not)

    def test_not_decorates_a_sequence(self):

        with build.build_not(build.build_sequence()) as not_:

            not_.task.tasks.append(ATask())
            not_.task.tasks.append(ATask())
        assert isinstance(not_.task, tasks.Sequence)
