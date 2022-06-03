# 一、为何要用知识图谱去增强预训练模型

1. 我们期望，预训练模型能够在需要特定知识或常识的任务中表现得更好，这就需要知道一些先验知识。
2. 为了提高fine-tune后的鲁棒性和可解释性。利用不同的知识来实现具有深刻理解和逻辑推理的预训练模型是不可缺少的。

# 二、常见的问题

1. 结构化信息的非结构化。如何将这样的信息与擅长提取非结构化文本信息的预训练模型结合起来，是开展工作的第一步。
2. 异构特征空间的对齐。由于token部分的特征与知识图谱的embedding是由两种不同的方法得到的，二者所处的特征空间是不完全相同的。
3. 知识噪声。知识图谱信息要与预训练模型良好融合才能提升效果。

# 三、预训练模型的分类

## 3.1 Token-based Pre-trained Models

基于相似上下文的词语可能拥有相同的语义，提出了CBOW和Skip-Gram去捕捉词语的语义相似性。GloVe捕捉词与词之间共现的统计数据，FastText用文本分类数据训练模型。这些模型简单而有效，但不能捕获多义词，只能获得固定的表示，被称为静态预训练模型。

---

## 3.2 Context-based Pre-trained Models

为了解决一词多义的问题，预训练模型需要去识别单词的语义并且在不同的上下文时动态生成单词的embedding。
[ELMO](./pretrain%20model/classical%20paper/ELMO.pdf)作为神经网络编码器，用双向语言模型，捕捉依赖于上下文的表示。但ELMO经常用于特征提取器去产生用于下游任务的模型的初始embedding，但模型剩下的参数不得不从头开始训练。

与此同时，[ULMFiT](./pretrain%20model/classical%20paper/ULMFiT.pdf)提出了通用语言模型微调，为模型提供了有价值的多阶段转移和微调技能。此外，transformer在机器翻译上，尤其是长文本依赖上展现了比LSTM好的性能。在这个背景下，OpenAI提出了[GPT](./pretrain%20model/classical%20paper/GPT.pdf), [GPT-2](./pretrain%20model/classical%20paper/GPT-2.pdf), [GPT-3](./pretrain%20model/classical%20paper/GPT-3.pdf)，使用了修改后的transformer的解码器作为语言模型学习通用表示。但受单向编码器的限制，GPT系列只能学习左边上下文，在生成任务上表现较好，但导致学习句子级别的语义没有达到最优。

为了克服单向的缺点，[BERT](./pretrain%20model/classical%20paper/BERT.pdf)采取了masked language modelling(MLM)，随机遮盖掉一部分单词，去根据上下文预测这单词本身，以及通过next sentence prediction(NSP)任务学习句子之间的语义表示。基于BERT，[RoBERTa](./pretrain%20model/classical%20paper/RoBERTa.pdf)设计了一些改进方案，包括更大批次、更长时间、动态遮盖等。[XLNet](./pretrain%20model/classical%20paper/XLNet.pdf)提出了一种基于排列语言建模的自回归方法（Permutation Language Model），随机选取一句话的一种排列，然后将末尾一定量的词给遮掩掉，通过自回归按这个排列依次预测被遮掩掉的词，通过单向方式来学习双向信息。

[T5](./pretrain%20model/classical%20paper/T5.pdf)不同于上面任意一种，而是采用了编码器-解码器（encoder-decoder）架构，无论是翻译，问答，还是分类等下游任务，都通过把数据转换为text-text格式来统一理解与生成两大任务。

[BART](./pretrain%20model/classical%20paper/BART.pdf)采用标准transformer-based架构，可以看作是BERT和GPT等很多最近的pretrain model的泛化。BART使用在Seq2Seq模块上额外加入了噪声函数，学习一个模型去重建原始的文本。最好的效果是使用sentence shuffling和text infilling。

![更多模型请看图片](./pretrain%20model/survey/qiu_pic1.png)
<center>邱锡鹏论文中对预训练模型和知识增强的预训练模型的总结</center>

---

# 四、知识增强的预训练模型