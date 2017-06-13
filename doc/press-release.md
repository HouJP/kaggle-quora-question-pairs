## YesOfCourse团队在Kaggle文本匹配竞赛中获得优异成绩

近日，实验室YesOfCourse团队（庞亮、范意兴、侯建鹏、岳新玉、牛国成）参加了Kaggle: Quora Question Pairs全球文本匹配算法竞赛并获得第4名，在华人参赛选手中排名第1。本次竞赛由全球最大的在线知识市场Quora举办，要求参赛队伍利用给定的标注数据，来推断平台用户提出的两个问题是否拥有相同的语义，并提供了包含约40万带有标签信息的文本数据集。

本次竞赛吸引了来自全球各地的3,307支团队参赛，包括世界各大顶级IT公司的数据科学家（如Airbnb、IBM、Microsoft等）以及各高校、研究机构的相关领域的研究者们。在华人参赛队伍中，包括来自清华、北大以及微软亚研等各高校和企业的参赛选手。

在本次文本匹配完成的任务中，YesOfCourse团队的LogLoss评价指标最终为0.11768，获得全球第4。在任务处理的过程中，包含了预处理、特征工程、模型构建、模型整合以及后处理五个关键步骤，构建了统计特征、NLP特征、图特征三大类特征集合，使用了Boosting、Linear、Deep Learning三种类型的模型，并提出了Deep Fusion的概念将模型的预测结果进行了整合。在比赛的过程中，团队开发了FeatWheel轻量级机器学习流程框架，从而简化特征融合、样本划分等机器学习任务。

最终的解决方案会整理发布在Github开源社区，附链接：[https://github.com/HouJP/kaggle-quora-question-pairs](https://github.com/HouJP/kaggle-quora-question-pairs)。

