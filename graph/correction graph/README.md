# Correction rate graphes

We experiment on the datasets wmt21.news.zh-en and wmt21.tedtalks.zh-en, and we test on our model, BERTScore, COMET-DA_2020, bleurt-21-beta, and Prism. 

For every source sentences, we generate machine translation pairs based on their human score difference (For example, group 1 contains the pairs with ). Then, we seperate these pairs into 25 groupes based on human score difference (For example, group 2 contains the pairs with human score difference less than 2 and greater than 1).

For every metrics, we computed how many pairs are correct with respect to the human score in each group and plot these graphes.
