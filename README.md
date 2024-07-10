# HERO: 基于语言风格的假新闻检测神经网络

本文档实现了论文《基于语言风格的假新闻检测神经网络》中提出的HERO模型。

## 环境需求
- PyTorch >= 1.9.1
- nltk >= 3.6.3

## 关于输入数据
1. 我们使用 **Story-based Fake** 数据集（gossipcop_v3-4_story_based_fake.json）来验证 HERO 模型的有效性。该数据集专注于假新闻的故事情节特征。此数据集由 gossipcop_v3_origin.json 文件生成。详细生成过程和特征描述请参考 [GitHub链接](https://github.com/junyachen/Data-examples)。

   数据集统计信息如下：
   - 总数：15729
   - 真新闻：11945
   - 假新闻：3784

2. 我们使用斯坦福的 **GloVe 100d** 词嵌入作为本论文中的词嵌入，文件名为 **glove.6B.100d.txt**。词嵌入文件可从 [Glove.6B.100d](https://nlp.stanford.edu/projects/glove/) 下载。

3. 为了处理数据集的 RST 和 CFG，我们使用以下网站的代码 [Generate RST and CFG tree](https://github.com/jiyfeng/DPLP)。

4. 这里我们提供了一个 **RST** 和 **CFG** 格式的简单示例，保存在文件夹 **/data/strtree_RST** 和 **/data/strtree_CFG** 中。我们为示例新闻 **Original_text_news_1.txt** 生成了 RST 和 CFG 树，最终在文件夹 **/data/strtree_RST** 和 **/data/strtree_CFG** 中分别生成了 news_1.txt (RST) 和 news_1.txt (CFG)。

## 复现结果
1. 当获取所有输入数据，包括词嵌入文件 **glove.6B.100d.txt**，文件夹 **/data/strtree_RST** 和 **/data/strtree_CFG** 后，我们可以使用以下命令结合结果文件夹中的训练模型，在数据集上复现结果。

```python
python test.py
```
