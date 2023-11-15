# Tokenization

> **时序数据上，可以考虑提出不同的tokenizer，实现每个token包含不同的长度，尤其对于类周期性信号**

- token : (单)词粒度 古典分词法
    **Sentence**: let's tokenize! Isn't this easy?
    **1.** | Let | 's | tokenize | ! | Is | n't | this | easy| ? |
    **2.** | Let's | tokenize! | Isn't | this | easy? |
    **3.** et. al.
    > - 对于未在词表中出现的词（Out Of Vocabulary, OOV ），模型将无法处理（未知符号标记为 [UNK]）
    > - 词表中的低频词/稀疏词在模型训无法得到训练（因为词表大小有限，太大的话会影响效率）。
    > - ⭐️ 很多语言难以用空格进行分词，例如英语单词的多形态，"look"衍生出的"looks", "looking", "looked"，其实都是一个意思，但是在词表中却被当作不同的词处理，模型也无法通过 old, older, oldest 之间的关系学到 smart, smarter, smartest 之间的关系。这一方面增加了训练冗余，另一方面也造成了大词汇量问题。

- character piece : 字(母)粒度
    **Sentence**: let's tokenize! Isn't this easy?
    **1.** | L | e | t | ' | s |  | t | o | k | e | n | i | z | e | .........
    > - 优点：词表很小（英文字母最多就是52个大小写字母 + 特殊字符）；鲁棒性强，没有OOV问题。
    > - 缺点：大部分单字母或者单字没有语义意义；根据词表将文本或句子转换为模型输入，其输入长度可能会很长，额外增加模型需要学习的参数，不仅使模型难以训练，训练也更耗时耗力。

- subword piece : 子词粒度
    **Word:** unfortunately  --> **Subword:** un + for + tun + ate + ly
    > - 通过一个有限的词表来解决所有单词的分词问题，同时尽可能将结果中 token 的数目降到最低。
    > - Subword 粒度在词与字符之间，能够较好的平衡 OOV 问题。
    - [BPE](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0), byte pair encoding GPT-1
        遵循的基本原则是：**从基本词汇表中选择组合在一起出现频次最高的两个符号中，并将其合成一个新的符号，加入基础词汇表，直到达到提前设定好的词汇量为止（词汇表大小是超参数）。**
        > - BPE 的优点就在于，可以很有效地平衡词典大小和编码步骤数（将语料编码所需要的 token 数量）。随着合并的次数增加，词表大小通常先增加后减小。迭代次数太小，大部分还是字母，没什么意义；迭代次数多，又重新变回了原来那几个词。所以词表大小要取一个中间值。
        > - BPE 的缺点, 对于同一个句子, 可能会有不同的 Subword 序列。不同的 Subword 序列会产生完全不同的 id 序列表示，这种歧义可能在解码阶段无法解决。在翻译任务中，不同的 id 序列可能翻译出不同的句子，这显然是错误的。
        > - 在训练任务中，如果能对不同的 Subword 进行训练的话，将增加模型的健壮性，能够容忍更多的噪声，而 BPE 的贪心算法无法对随机分布进行学习。
        > - BPE 一般适用在欧美语言拉丁语系中，因为欧美语言大多是字符形式，涉及前缀、后缀的单词比较多。而中文的汉字一般不用 BPE 进行编码，因为中文是字无法进行拆分。对中文的处理通常只有分词和分字两种。
    - Byte-level BPE（BBPE） GPT2/LLAMA
        - BBPE核心思想将BPE的从字符级别扩展到子节（Byte）级别
    - [WordPiece](https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7) BERT
        - **The only difference between WordPiece and BPE is the way in which symbol pairs are added to the vocabulary. At each iterative step, WordPiece chooses a symbol pair which will result in the largest increase in likelihood upon merging. (例如拆分前后的信息增益/似然概率)**
        - WordPiece和BPE的区别就在每次merge的过程中， BPE是通过合并最高频次的，而WordPiece是选择让似然概率最大的值，具体的计算使用合并后的概率值，除以合并前的概率值
        - 你一定会想，这样的计算是非常复杂的，确实是这样的，每次选取两个token组合的时候我们都需要测试所有的可能性并且计算句子的困惑度，所以时间复杂度是O(k^2) ,其中K表示的当前词表的大小。但是在原论文中有些简单的小技巧可以加速计算，比如一次组合多个词，一次性算困惑度等
    - Unigram UniLM
        - 从网络结构上看，UniLM它的结构是和BERT相同的编码器的结构。但是从它的预训练任务上来看，它不仅可以像自编码语言模型那样利用掩码标志的上下文进行训练，还可以像自回归语言模型那样从左向右的进行训练。甚至可以像Encoder-Decoder架构的模型先对输入文本进行编码，再从左向右的生成序列。
        - UniLM采用了Unigram的方式对其进行了分词
        - 它和 BPE 以及 WordPiece 从表面上看一个大的不同是，前两者都是初始化一个小词表，然后一个个增加到限定的词汇量，而 Unigram Language Model 却是先初始一个大词表，接着通过语言模型评估不断减少词表，直到限定词汇量。
    - SentencePiece XLNet
        - 唯一Token数量是预先确定的
            与大多数假设无限词汇量的无监督分词算法不同，SentencePiece 在训练分词模型时，使最终的词汇表大小固定，例如：8k、16k 或 32k。
        - 从原始句子进行训练
            以前的子词（sub-word）实现假设输入句子是预标记（pre-tokenized）的。 这种约束是有效训练所必需的，但由于我们必须提前运行依赖于语言的分词器，因此使预处理变得复杂。 SentencePiece 的实现速度足够快，可以从原始句子训练模型。 这对于训练中文和日文的tokenizer和detokenizer很有用，因为在这些词之间不存在明确的空格。
        - 空格被视为基本符号
        - 子词正则化和 BPE-dropout