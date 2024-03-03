from dachi.comm import Post, Request
from dachi.comm._query import Post


class DummyQuery(Request):

    def __init__(self):
        super().__init__()
        self.called = False
    
    def post(self, post: Post):

        print('Calling')
        self.called = True
        post.response = 'RESPONDED'


class TestQuest:

    def test_query_prepares_post(self):

        query = DummyQuery()
        post = Post()
        post.request(query)
        assert query.called is True

    def test_post_is_called(self):

        query = DummyQuery()
        post = Post()
        post.request(query)
        assert query.called is True

    def test_content_is_correct_after_calling(self):

        query = DummyQuery()
        post = Post()
        post.request(query)
        assert post.response == "RESPONDED"


# class TestSignal:

#     def test_query_prepares_post(self):

#         signal = DummySignal()
#         signal.register(signal.on_post)
#         signal.post(Request('signal'))
#         assert signal.called is True

#     def test_post_is_called(self):

#         signal = DummySignal()
#         request = Request('signal')
#         request.on_post = signal.on_post
#         signal.post(request)
#         assert signal.called is True



# class DummySignal(Signal):

#     def __init__(self):
#         super().__init__()
#         self.called = False
#         self.post_it = False

#     def prepare_response(self, request):
#         return 'respond!'
    
#     def prepare_post(self, request):

#         self.post_it = True

#     def on_post(self, request):
#         self.called = True
