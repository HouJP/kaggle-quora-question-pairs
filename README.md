****

#	<center>Kaggle: Quora Question Pairs</center>
##### <center>Author: Liang Pang, Yixing Fan, Jianpeng Hou, Xinyu Yue, Guocheng Niu</center>

****

##	Categories
*	[Overview](#overview)

****

##	<a name="overview">Overview</a>

First congratulations to every participants and thanks a lot to the organizers. I'll make a overview of our solution. 

We can simply divide the solution into different parts: Pre-processing, Feature Engineering, Modeling and Post-processing.

#### Pre-processing

We made some different versions of original data (train.csv & test.csv).

1.	`Text-cleaning`: spell correction & symbol processing & acronyms restore & ...
2. `Word-stemming`: use SnowballStemmer & ...
3. `Shared-word-removing`: delete the words appeared in the both sides

#### Feature Engineering

There was around 1400+ features in the Feature Pool which will be combined in different ways. These features can be classified as the following categories.

1.	`Statistic`: rate of shared words & length of sentences & number of words & ...
2.	`NLP`: analysis of grammar tree & negative words count & ...
3. `Graph`: pagerank & hits & shortest path & ...

#### Modeling

We used `DL Models` & `XGB` & `LGB` & `LR`. The best single model scored about 0.122~0.124 on the LB. We build a multi-layer stacking system to ensemble different models together (about 140+ model results), this method can get a gain of ~0.007 on public LB.

#### Post-processing

As we all knonw, the distribution of the training data and test data were quite different. We cutted the data into different parts  according to the clique size and rescale the results in different parts, this method can get a gain of ~0.001 on public LB. 

What's more, we developed a light weight framework 'FeatWheel' to help us to finish ML jobs, such as feature extraction & feature merging, you may enjoy it.

****


