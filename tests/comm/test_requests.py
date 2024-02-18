from dachi.comm import Query, Signal, Request


class DummyQuery(Query):

    def __init__(self):
        super().__init__()
        self.called = False
        self.response_called = False

    def prepare_response(self, request):
        return 'respond!'
    
    def exec_post(self, request):
        self.respond(request, 'CALLED')

    def on_post(self, request):
        self.called = True

    def on_response(self, request):
        self.response_called = True


class DummySignal(Signal):

    def __init__(self):
        super().__init__()
        self.called = False
        self.post_it = False

    def prepare_response(self, request):
        return 'respond!'
    
    def prepare_post(self, request):

        self.post_it = True

    def on_post(self, request):
        self.called = True


class TestQuery:

    def test_query_prepares_post(self):

        query = DummyQuery()
        query.register(query.on_post)
        query.post(Request('query'))
        assert query.called is True

    def test_post_is_called(self):

        query = DummyQuery()
        request = Request('query')
        request.on_post = query.on_post
        query.post(request)
        assert query.called is True

    def test_respond_is_called_after_post(self):

        query = DummyQuery()
        request = Request('query')
        request.on_response = query.on_response
        query.post(request, False)
        assert query.response_called is True


class TestSignal:

    def test_query_prepares_post(self):

        signal = DummySignal()
        signal.register(signal.on_post)
        signal.post(Request('signal'))
        assert signal.called is True

    def test_post_is_called(self):

        signal = DummySignal()
        request = Request('signal')
        request.on_post = signal.on_post
        signal.post(request)
        assert signal.called is True
