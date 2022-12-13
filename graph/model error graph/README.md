# MT model error graphes

We experiment on the datasets wmt21.news.zh-en and wmt21.tedtalks.zh-en, and we test on COMET-DA_2020.

For every source sentences, we generate machine translation pairs based on their human score difference. Then, we seperate these pairs into 25 groupes based on human score difference (For example, group 2 contains the pairs with human score difference less than 2 and greater than 1).

We computed how many pairs are incorrect with respect to the human score in each group and we record their MT models. Therefore, for certain MT models, we plot how many times COMET incorrectly evaluate it with respect to different human score difference.
