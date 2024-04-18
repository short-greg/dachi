# from dachi.comm import Post, Request
# from dachi.comm._query import Post
# import time

# class DummyQuery(Request):

#     def __init__(self):
#         super().__init__()
#         self.called = False
    
#     def post(self, post: Post):

#         self.called = True
#         post.response = 'RESPONDED'


# class TestRequest:

#     def test_query_prepares_post(self):

#         query = DummyQuery()
#         post = Post()
#         post.request(query)
#         assert query.called is True

#     def test_post_is_called(self):

#         query = DummyQuery()
#         post = Post()
#         post.request(query)
#         assert query.called is True

#     def test_content_is_correct_after_calling(self):

#         query = DummyQuery()
#         post = Post()
#         post.request(query)
#         assert post.response == "RESPONDED"

#     # # Test the threaded version turned off cause it is slow
#     # def test_content_is_correct_after_calling_threaded(self):

#     #     query = DummyQuery()
#     #     post = Post()
#     #     post.threaded_request(query)
#     #     time.sleep(1)
#     #     assert post.response == "RESPONDED"
