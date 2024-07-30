# -*- coding: utf-8 -*-
"""
기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 분류하기
**************************************************************
**Author**: `Sean Robertson <https://github.com/spro>`_

우리는 단어를 분류하기 위해 기본 문자 수준의 순환 신경망(RNN)을 구축하고 훈련할 것입니다.
이 튜토리얼과 다른 두 개의 자연어 처리(NLP) "기초부터 시작하는" 튜토리얼은 모두 데이터를 전처리하고 NLP 모델링 하는 방법을 보여줍니다.
특히 이 튜토리얼은 `torchtext`의 편의 기능을 많이 사용하지 않아, NLP 전처리가 낮은 수준에서 어떻게 작동하는지 볼 수 있습니다.

문자 수준 RNN은 단어를 일련의 문자로 읽어 각 단계에서 예측과 "은닉 상태"를 출력하고, 이전 은닉 상태를 다음 단계에 입력합니다.
우리는 최종 예측을 출력으로 사용합니다. 즉, 단어가 속한 클래스입니다.

구체적으로, 우리는 18개 언어의 몇 천 개의 성씨로 훈련하고, 철자를 기반으로 이름의 출신 언어를 예측할 것입니다.

.. code-block:: sh

    $ python predict.py Hinton
    (-0.47) Scottish
    (-1.52) English
    (-3.57) Irish

    $ python predict.py Schmidhuber
    (-0.19) German
    (-2.48) Czech
    (-2.68) Dutch


준비 사항
=======================

이 튜토리얼을 시작하기 전에 PyTorch를 설치하고 Python 프로그래밍 언어와 텐서에 대한 기본적인 이해가 필요합니다:

-  https://pytorch.org/ 설치 지침
-  :doc:`/beginner/deep_learning_60min_blitz` PyTorch를 시작하고 텐서의 기본을 배우기 위해
-  :doc:`/beginner/pytorch_with_examples` 폭넓고 깊이 있는 개요
-  :doc:`/beginner/former_torchies_tutorial` Lua Torch 사용자를 위한 튜토리얼

또한 RNN과 그 작동 방식에 대해 알고 있으면 유용합니다:

-  `The Unreasonable Effectiveness of Recurrent Neural Networks <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>`__ 다양한 실제 예시를 보여줍니다.
-  `Understanding LSTM Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__ LSTM에 관한 것이지만 RNN 전반에 대해 유익합니다.

데이터 준비
==================

.. note::
   데이터를 `여기 <https://download.pytorch.org/tutorial/data.zip>`_ 에서 다운로드하고 현재 디렉터리에 압축을 푸십시오.

``data/names`` 디렉터리에는 ``[Language].txt``라는 이름의 18개 텍스트 파일이 포함되어 있습니다. 각 파일에는 한 줄에 하나씩 이름이 들어 있으며, 대부분은 로마자로 표기되어 있습니다(하지만 여전히 Unicode를 ASCII로 변환해야 합니다).

우리는 결국 언어별 이름 목록의 딕셔너리 ``{language: [names ...]}``를 얻게 될 것입니다. 나중의 확장성을 위해 범용 변수 "category"와 "line"(우리의 경우 언어와 이름)을 사용합니다.
"""
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 유니코드 문자열을 ASCII로 변환하기, https://stackoverflow.com/a/518232/2809427 참조
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# category_lines 딕셔너리 구축, 언어별 이름 목록
category_lines = {}
all_categories = []

# 파일을 읽고 라인으로 분할
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


######################################################################
# 이제 각 category (언어)를 line (이름)에 매핑하는 사전인 ``category_lines`` 를 만들었습니다.
# 나중에 참조할 수 있도록 ``all_categories`` (언어 목록)와 ``n_categories`` 도 추적합니다.
#

print(category_lines['Italian'][:5])


######################################################################
# 이름을 텐서로 변환하기
# --------------------------
#
# 이제 모든 이름이 정리되었으므로 이를 텐서로 변환해야 합니다.
#
# 단일 문자를 나타내기 위해 크기 ``<1 x n_letters>``의 "원-핫 벡터"를 사용합니다.
# 원-핫 벡터는 현재 문자의 인덱스에 1을 제외한 모든 값이 0으로 채워져 있습니다. 예: ``"b" = <0 1 0 0 0 ...>``.
#
# 단어를 만들기 위해 이러한 벡터를 2D 행렬 ``<line_length x 1 x n_letters>``로 결합합니다.
#
# 추가 1차원은 PyTorch가 모든 것을 배치로 처리한다고 가정하기 때문입니다. 여기서는 배치 크기 1을 사용하고 있습니다.
#

import torch

# 모든 문자의 인덱스 찾기, 예: "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# 데모를 위해, 문자를 ``<1 x n_letters>`` 텐서로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 라인을 ``<line_length x 1 x n_letters>`` 텐서로 변환
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())


######################################################################
# 네트워크 생성
# ====================
#
# 자동 미분 이전에는 Torch에서 순환 신경망을 생성하기 위해 여러 타임스텝에 걸쳐 레이어의 매개변수를 복제해야 했습니다.
# 레이어는 은닉 상태와 그래디언트를 보유했으며, 이는 이제 그래프 자체에서 완전히 처리됩니다.
# 이는 RNN을 매우 "순수한" 방식으로 구현할 수 있음을 의미합니다. 일반 피드포워드 레이어처럼 말이죠.
#
# 이 RNN 모듈은 "바닐라 RNN"을 구현하며 입력과 은닉 상태에 대해 작동하는 3개의 선형 레이어와 출력 후 ``LogSoftmax`` 레이어로 구성됩니다.
#

import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


######################################################################
# 이 네트워크의 단계를 실행하려면 입력(이 경우 현재 문자의 텐서)과 이전 은닉 상태를 전달해야 합니다(처음에는 0으로 초기화).
# 우리는 각 언어의 확률을 출력하고 다음 은닉 상태를 얻습니다(다음 단계에 사용).
#

input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)


######################################################################
# 효율성을 위해 각 단계마다 새로운 텐서를 생성하고 싶지 않으므로 ``letterToTensor`` 대신 ``lineToTensor``를 사용하고 슬라이스를 사용합니다.
# 이것은 사전 계산된 텐서 배치를 사용하여 더 최적화할 수 있습니다.
#


input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)


######################################################################
# 출력은 ``<1 x n_categories>`` 텐서로, 각 항목은 해당 범주(카테고리)의 가능성을 나타냅니다 (값이 클수록 가능성이 높음).
#


######################################################################
#
# 훈련
# ========
# 훈련 준비
# ----------------------
#
# 훈련에 들어가기 전에 몇 가지 도우미 함수를 만들어야 합니다.
# 첫 번째는 네트워크의 출력을 해석하는 함수로, 이는 각 카테고리의 가능성을 나타냅니다.
# 우리는 ``Tensor.topk``를 사용하여 가장 큰 값의 인덱스를 얻을 수 있습니다:
#

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))


######################################################################
# 또한 훈련 예제(이름과 그 언어)를 빠르게 얻는 방법이 필요합니다:
#

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)


######################################################################
# 네트워크 훈련
# --------------------
#
# 이제 이 네트워크를 훈련시키려면 많은 예제를 보여주고, 추측하게 하며, 틀렸을 경우 알려주면 됩니다.
#
# 손실 함수로는 ``nn.NLLLoss``가 적합한데, 이는 RNN의 마지막 레이어가 ``nn.LogSoftmax``이기 때문입니다.
#

criterion = nn.NLLLoss()


######################################################################
# 각 훈련 루프는 다음을 수행합니다:
#
# - 입력과 타겟 텐서 생성
# - 초기 은닉 상태를 0으로 초기화
# - 각 문자를 읽으면서
#
#    - 다음 문자를 위해 은닉 상태 유지
#
# - 최종 출력을 타겟과 비교
# - 역전파 수행
# - 출력과 손실 반환
#

learning_rate = 0.005 # 너무 높으면 발산할 수 있고, 너무 낮으면 학습하지 않을 수 있음

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 매개변수의 그래디언트를 값에 더하고, 학습률을 곱함
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


######################################################################
# 이제 많은 예제를 사용하여 이를 실행하기만 하면 됩니다.
# ``train`` 함수는 출력과 손실을 모두 반환하므로 추측을 출력하고 손실을 추적할 수 있습니다.
# 예제가 수천 개이므로 ``print_every`` 예제마다 출력하고, 손실의 평균을 계산합니다.
#

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# 손실을 추적하기 위해
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # ``iter`` 번호, 손실, 이름 및 추측을 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 손실 평균을 손실 목록에 추가
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


######################################################################
# 결과 플로팅
# --------------------
#
# ``all_losses``의 과거 손실을 플로팅하면 네트워크가 학습하는 모습을 볼 수 있습니다:
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


######################################################################
# 결과 평가
# ======================
#
# 네트워크가 다른 카테고리에서 얼마나 잘 수행되는지 보기 위해 혼동 행렬을 생성할 것입니다.
# 실제 언어(행)별로 네트워크가 추측한 언어(열)를 나타냅니다.
# 혼동 행렬을 계산하기 위해 여러 샘플을 ``evaluate()`` 함수로 실행합니다. 이는 ``train()`` 함수에서 역전파 부분만 제외한 것과 같습니다.
#

# 혼동 행렬에서 올바른 추측을 추적
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# 주어진 라인에 대해 출력만 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 여러 예제를 살펴보고 올바르게 추측한 항목을 기록
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 각 행을 합으로 나누어 정규화
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 플롯 설정
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# 각 축에 레이블 강제 설정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


######################################################################
# 주축에서 벗어난 밝은 점을 보면 네트워크가 잘못 추측한 언어를 알 수 있습니다.
# 예를 들어, 중국어와 한국어, 스페인어와 이탈리아어 등.
# 그리스어는 매우 잘 추측하고, 영어는 매우 잘못 추측합니다(아마도 다른 언어와 겹치기 때문일 것입니다).
#


######################################################################
# 사용자 입력으로 실행
# ---------------------
#

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')


######################################################################
# Practical PyTorch 저장소의 최종 버전은 위의 코드를 몇 개의 파일로 분할합니다:
#
# -  ``data.py`` (파일 로드)
# -  ``model.py`` (RNN 정의)
# -  ``train.py`` (훈련 실행)
# -  ``predict.py`` (명령줄 인수로 ``predict()`` 실행)
# -  ``server.py`` (``bottle.py``로 JSON API로 예측 제공)
#
# ``train.py``를 실행하여 네트워크를 훈련시키고 저장하십시오.
#
# ``predict.py``를 실행하여 이름을 예측 결과로 보기:
#
# .. code-block:: sh
#
#     $ python predict.py Hazaki
#     (-0.42) Japanese
#     (-1.39) Polish
#     (-3.51) Czech
#
# ``server.py``를 실행하고 http://localhost:5533/Yourname 를 방문하여 예측 결과를 JSON 출력으로 받습니다.
#


######################################################################
# 연습 문제
# =========
#
# -  다른 데이터셋으로 시도해 보세요. 예를 들어:
#
#    -  단어 -> 언어
#    -  이름 -> 성별
#    -  캐릭터 이름 -> 작가
#    -  페이지 제목 -> 블로그 또는 서브레딧
#
# -  더 크고/더 나은 형태의 네트워크로 더 나은 결과를 얻어 보세요.
#
#    -  더 많은 선형 레이어 추가
#    -  ``nn.LSTM`` 및 ``nn.GRU`` 레이어 시도
#    -  이러한 RNN을 여러 개 결합하여 상위 레벨 네트워크로 구성
#
