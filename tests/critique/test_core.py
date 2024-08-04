from dachi.critique import _core
from ..core.test_core import SimpleStruct


class TestEvaluation:

    def test_evaluation_has_the_evaluations(self):

        evaluation = _core.Evaluation(
            values=[{
                'mse': 'Great'
            }]
        )
        assert evaluation.values[0]['mse'] == 'Great'


class TestTextualCriterion:

    def test_out_result_is_correct(self):

        criterion = _core.TextualCriterion(
            name='Accuracy',
            desc='Evaluate how close the output is to the target'
        )
        assert criterion.out_format()['Accuracy'] == "<Result>"

    def test_criteria_is_correct(self):

        criterion = _core.TextualCriterion(
            name='Accuracy',
            desc='Evaluate how close the output is to the target'
        )
        assert criterion.criteria()['Accuracy'] == criterion.desc


class TestHeaderView:

    def test_header_view(self):

        x = SimpleStruct(x='hi')
        t = SimpleStruct(x='bye')

        criterion = _core.TextualCriterion(
            name='Accuracy',
            desc='Evaluate how close the output is to the target'
        )
        view = _core.HeaderView(
            criterion
        )
        text = view(
            x, t
        )
        print(text)
        # TODO: test the output
        assert False



# class TestSample:

#     def test_sample_renders_the_data(self):
        
#         sample = _core.Sample(
#             data={'x': SimpleStruct(x='2')}
#         )
#         assert sample.data['x'].x == '2'


# class TestBatch:

#     def test_batch_loads_the_data(self):
        
#         batch = _core.Batch(
#             data=[
#                 {'x': SimpleStruct(x='2')},
#                 {'x': SimpleStruct(x='4')}
#             ])
#         batch.data[0]
#         assert batch.data[0]['x'].x == '2'

#     def test_batch_loads_the_data_from_samples(self):
        
#         sample1 = _core.Sample(
#             data={'x': SimpleStruct(x='2')}
#         )
#         sample2 = _core.Sample(
#             data={'x': SimpleStruct(x='3')}
#         )

#         batch = _core.Batch.from_samples(
#             [sample1, sample2]
#         )
#         batch.data[0]
#         assert batch.data[0]['x'].x == '2'


#     def test_batch_loads_the_data_from_lists(self):
    
#         batch = _core.Batch.create(
#             x=[SimpleStruct(x='2'), SimpleStruct(x='3')]
#         )
#         batch.data[0]
#         assert batch.data[0]['x'].x == '2'


# class TestSupervised:

#     def test_supervised_out_format_contains_name(self):
#         supervised = _core.Supervised(
#             name='X', how='Evaluate .'
#         )
#         assert 'values' in supervised.criteria()[supervised.name]
