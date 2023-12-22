import sys
import os
from examples.vocab_learning import app

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # print(os.getcwd() + 'examples/vocab_learning/')
    # sys.path.append(os.curdir + 'examples/vocab_learning/')
    teacher = app.TeacherApp()
    teacher.run()