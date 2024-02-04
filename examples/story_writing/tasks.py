from dachi.behavior._status import SangoStatus
from dachi.storage import Prompt
from ..tools.base import ProcessResponse, Processed
import typing

from dachi.storage import Conv, R

from dachi.behavior import Action


QUESTION_PROMPT = Prompt(
["summary"],
"""
# 役割
あなたはストーリー作成のアシスタントです。作家が物語の各ステップを理解するのを手伝ってください。
今回のステップが物語の問いかけを決めるステップです。

# 現時点の会話の要約
{summary}

# 指示
1. はじめに、適切な挨拶をしてください。
2. 作家の返答に基づいて、作家が物語の中心となる問いを定義するのを手伝ってください。中心となる問いは、かなり正確で答えにくいものでなければなりません。そうでなければ、興味深い話を書くことは難しくなります。例えば、「世界に一人きりでも人間は生き残ることができるか」というのは良い質問です。「自由とは何か」というのは広すぎます。「子犬を蹴るのは大丈夫か」というのは答えやすくて面白くありません。作家に、その答えをどのように改善できるかについてアドバイスを与えてください。
3. 問いが決まったら、"完了"だけと返答してください。
"""
)

ANSWER_PROMPT = Prompt(
["summary"],
"""
# 役割
あなたはストーリー作成のアシスタントです。作家が物語の各ステップを理解するのを手伝ってください。ここに物語を定義する現在の進行状況の要約があります。
今回のステップが問い掛けの可能な答えを決めるステップです。

# 概要
{summary}

# 指示
1. はじめに、適切な挨拶をしてください。
2. 作家の返答に基づいて、次に、作家が物語の中心となる質問への答えを考え出すのを手伝ってください。その後、中心となる質問への可能な答えを考え出すのを手伝ってください。それぞれの答えは妥当であり、物語のいずれかの時点で登場人物や特性を代表するものでなければなりません。作家はその答えが正しい理由について確固たる根拠を提供しなければなりません。
3. 可能な答えをいくつかだしたら、"完了"だけと返答してください。
"""
)

SUMMARY_PROMPT = Prompt(
["summary", "conversation"],
"""
# 説明
あなたはストーリー作成のアシスタントです。作家が物語のアイデアを思いつくための現在の進捗を要約してください。以下に以前の要約が示されています。"新しい会話"を使用して要約を更新してください。

# 以前のようやく
{summary}

# 新しい会話
{conversation}
"""
)

DEFAULT_PROMPT = Prompt(
["summary"],
"""
# 説明
あなたは物語作成のアシスタントです。作家が物語のメインテーマを ===Theme=== で考えるのを手伝いました。

# 指示
まずは、適切に挨拶してください。
作家と一緒に ===Theme=== についてさらに議論してください。

===Theme===
{summary}
"""
)


class ProcessComplete(ProcessResponse):

    def __init__(self, to_set: str, state: typing.Dict):

        self.state = state
        self.to_set = to_set

    def process(self, content) -> Processed:
        
        if content == '完了':
            self.state[self.to_set] = True
            return Processed(
                '', False, True
            )
        
        self.state[self.to_set] = False
        
        return Processed(
            content, True, False
        )


class PrintOut(Action):

    def __init__(self, output: str, r: R):
        super().__init__()
        self.output = output
        self.r = r

    def act(self) -> SangoStatus:
        print(self.output, self.r())
        return self.SUCCESS


# class StoryTellerConv(ChatConv):
    
#     def __init__(self, max_turns: int=None):

#         # add introductory message
#         super().__init__(
#             max_turns
#         )
#         self._completed = False
#         self._conv_turns: typing.List[Message] = []

#     def add_turn(self, role: str, text: str) -> Conv:
#         if role == 'assistant':
#             if text == '完了':
#                 self._completed = True
#             else:
#                 self._completed = False
#                 super().add_turn(role, text)
#                 self._conv_turns.append(self._turns[-1])
#         elif role == 'user':
#             self._completed = False
#             super().add_turn(role, text)
#             self._conv_turns.append(self._turns[-1])
#         else:
#             super().add_turn(role, text)

#     def reset(self):
#         super().reset()
#         self._completed = False
#         self._conv_turns.clear()

#     def as_text(self, heading: str = None) -> str:
#         return super().as_text(heading)

#     @property
#     def conv(self) -> str:

#         result = ''
#         for turn in self._conv_turns:
#             result += f'{turn.as_text()}\n'
#         return result

#     @property
#     def completed(self) -> bool:
#         return self._completed


# class SummaryCompletion(ChatConv):
    
#     def __init__(self, max_turns: int=None):

#         # add introductory message
#         super().__init__(
#             max_turns
#         )
#         self._summary = None

#     def add_turn(self, role: str, text: str) -> Conv:
#         if role == 'assistant':
#             self._summary = text
#             super().add_turn(role, text)
#         else:
#             super().add_turn(role, text)

#     @property
#     def summary(self):
#         return self._summary

#     def reset(self):
#         super().reset()
#         self._completed = False