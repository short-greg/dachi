import chromadb
import dbm
from chromadb.config import Settings
from .. import tako
from ..tako import nodemethod, Field, FieldList
from chromadb.utils import embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002"
            )

# https://gpt-index.readthedocs.io/en/latest/examples/low_level/ingestion.html
# https://pypi.org/project/chromadb/

# https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide

student_info = """
Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.
"""

club_info = """
The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.
"""

university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world."""

class ChromaDBIndex:

    def __init__(self):

        client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet",
            persist_directory="db/")
        )
        students_embeddings = openai_ef([student_info, club_info, university_info])

        collection = client.create_collection(name="Students")
        collection.add(
            embeddings = students_embeddings,
            documents = [student_info, club_info, university_info],
            metadatas = [{"source": "student info"},{"source": "club info"},{'source':'university info'}],
            ids = ["id1", "id2", "id3"]
        )
        self.collection = collection
        self.idx = 4

    @nodemethod(['query'], ['documents'])
    def retrieve(self, query: str=None):
        
        if query is None:
            return []
        results = self.collection.query(query_texts=query)
        return results['documents']

    @nodemethod(['document'], ['document'])
    def add(self, document: str=None, source: str=None):

        if document is None:
            return document
        document_embeddings = openai_ef([document])
        self.collection.add(
            embeddings = document_embeddings,
            documents = [document],
            metadatas = [{"source": source}],
            ids = [f"id{self._idx}"]
        )


class PromptCompletor:

    def __init__(self):
        pass

    
    @nodemethod(['prompt'], ['completion'])
    def retrieve(self, prompt: str=None) -> str:

        pass

