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

![更多模型请看图片](./image/README/qiu_pic1.png)

<center>邱锡鹏论文中对预训练模型和知识增强的预训练模型的总结</center>

---

# 四、知识增强的预训练模型

## 4.1 知识表示

### 4.1.1 基于距离度量的传统模型

知识表示学习用深度学习的方式学习实体和关系在知识图（knowledge graph）中的表示，有效地测量了实体和关系之间的语义相关性，并缓解了稀疏性问题。传统的知识表示模型，例如[TransE](./knowledge%20injection/classical%20paper/NIPS-2013-translating-embeddings-for-modeling-multi-relational-data-Paper.pdf)，基于距离度量的方法，实体h和实体t被转换向量r关联起来:
$h+r≈t$，从而可以用$(h,r,t)$三元组来表达知识。但TransE对于复杂关系建模效果差，以及没有充分利用知识库中的额外信息。

### 4.1.2 语义匹配模型

语义匹配模型将实体的潜在语义和关系与基于相似性的评分函数来进行匹配。这类模型将每个实体与向量关联，将关系与矩阵关联，一个三元组$(h,r,t)$的分数通过双线性函数或者神经网络等来评估。

### 4.1.3 图神经网络模型

上面的模型都基于三元组来编码实体和关系，图神经网络基于整个图结构来做。图卷积神经网络（[GCN](./knowledge%20injection/classical%20paper/Spectral%20Networks%20and%20Deep%20Locally%20Connected%20Networks%20on%20Graphs.pdf)）聚合了每个节点的邻居信息，有效地表示了节点。[R-GCN](./knowledge%20injection/classical%20paper/R-GCN%20Modeling%20Relational%20Data%20with%20Graph%20Convolutional%20Networks.pdf)针对link prediction和entity classification两个任务，用图卷积神经网络建模关系型数据，解决了知识库中高维关系型数据的特征表示。[SACN](./knowledge%20injection/classical%20paper/End-to-end%20Structure-Aware%20Convolutional%20Networks%20for%20Knowledge%20Base%20Completion.pdf)
用了端到端的学习框架，编码器利用了图节点结构和属性，解码器简化了[ConvE](./knowledge%20injection/classical%20paper/Convolutional%202D%20Knowledge%20Graph%20Embeddings.pdf)并保持了TransE的平移不变性。[Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](./knowledge%20injection/classical%20paper/Learning%20Attention-based%20Embeddings%20for%20Relation%20Prediction%20in%20Knowledge%20Graphs.pdf)基于[GAT](./knowledge%20injection/classical%20paper/GAT%20GRAPH%20ATTENTION%20NETWORKS.pdf)的框架，提出了一种基于注意力的特征嵌入，该嵌入可以捕获编码器中的实体和关系特征，根据邻居节点的重要度来分配不同权重。[CompGCN](./knowledge%20injection/classical%20paper/CompGCN%20COMPOSITION-BASED%20MULTI-RELATIONAL%20Graph%20Convolutional%20Networks.pdf)认为在消息传递过程中，应该整体考虑节点和关系，利用知识图嵌入技术中各种关系的组合操作，并根据关系的数量来共同嵌入节点和关系。

## 4.2 知识增强的预训练模型

这一类模型可以提升对于语义的深度理解和逻辑归因，从而提升预训练模型在下游任务的表现。我们可以把它分类为：实体增强(entity enhanced)、三元组增强(triplet enhanced)和其他知识(other knowledge enhanced)增强的预训练模型。
还有一个标准是couple-based和decouple-based（retrieval-based）。entity enhanced均属于couple-based。

![](./image/README/knowledge%20enhanced%20pretrain%20model.png)

entity enhanced: 注重实体知识，通过注入实体信息和语言表示来注入知识。一种是把实体特征和单词嵌入通过文本编码器注入，另一种是发挥编码器的力量在知识图的监督下去捕捉隐含的文本知识。

triplet enhanced: 注重triplet层面的知识，使用了实体的特征和三元组的结构信息，通过注入或者检索triplet来注入知识。但coupled-based无法保证符号性知识的可解释性，因为这一类模型用了同一组参数存储知识和语言信息。decoupled-based模型使得知识利用可检测可解释。

other knowledge enhanced: 通过联合训练知识和序列或从所提供的语料库中检索相关信息来实现性能的提高。

### 4.2.1 entity enhanced pre-trained models

---

**[SenseBERT](./knowledge%20injection/classical%20paper/SenseBERT%20Driving%20Some%20Sense%20into%20BERT.pdf):** 在词义层面应用自监督，显著提升词义消歧能力，在复杂的 Word in Context (WiC) 语言任务中取得了当前最优结果。

Method: 添加一个掩蔽词义预测任务作为辅助任务，添加了一个专家构建的外部知识库WordNet，训练模型除了预测mask的单词本身以外，还要预测单词的语义，这是一个分类任务。WordNet把词语分成了45个超类，并且把类别映射到和word embedding同样的空间。

![](image/README/SenseBERT.png)

评价：起到了一定的消除歧义作用，但把词语的SuperSense分为45类，这样的知识库形式过于简单，如果复杂知识嵌入可能难以参考借鉴。并且，S直接映射到和word同样的空间，把S和W值相加，会不会降低了可解释性。

---
**[SemtiLARE](./knowledge%20injection/classical%20paper/SentiLARE%20Sentiment-Aware%20Language%20Representation%20Learning%20with%20Linguistic%20Knowledge.pdf):** 解决细粒度的情感分类，引入单词级别的语言学外部知识，通过标签感知的MLM任务为BERT注入情绪极性及其词性。

Method: 和上面那个研究很像，只不过这次加入的是单词级别的情感编码。通过一个词在这个句子中的词性和上下文，可以得到一个对这个词在此处情感的打分，最终标记为positive/negative/neutral，也就是极性。然后把词性的编码和极性的编码都加入到embedding中，预测的时候也是预测单词、词性、极性三样，loss相加。


![](./image/README/SentiLARE.png)

评价：本质上是利用了一个外部的[SentiWordNet](https://github.com/aesuli/SentiWordNet)检索词性和极性，然后再把检索出来的结果融入embedding中。把情感的强弱忽略，只有三个标签：正、负、中性，词性的编码也只有5种，问题和上面的一样，可能过于简单，并且只适用于非常简单的外部知识形式。但对于有一个SentiWordNet网络作为外部知识的情况，会不会不转变成单个的标签，而是直接使用网络输出的编码会好一点？

---


