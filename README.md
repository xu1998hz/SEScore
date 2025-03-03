![Alt text](image/logo_sescore.png?raw=true "Sescore logo")

# SEScore
## Best unsupervised evaluation metric in WMT22 in all language pairs and domains! 


This repo contains all the codes for SEScore implementation. SEScore is a reference-based text-generation evaluation metric that requires no pre-human-annotated error data, described in our paper [Not All Errors are Equal: Learning Text Generation Metrics using Stratified Error Synthesis.](https://arxiv.org/abs/2210.05035) from EMNLP 2022. Reader can refer https://research.google/pubs/pub51897/ for our WMT22 results!

Its effectiveness over prior methods like BLEU, BERTScore, BARTScore, PRISM, COMET and BLEURT has been demonstrated on a diverse set of language generation tasks, including translation, captioning, and web text generation. [Readers have even described SEScore as "one unsupervised evaluation to rule them all"](https://twitter.com/LChoshen/status/1580136005654700033) and we are very excited to share it with you!


<h3>Install all dependencies:</h3>

````
pip install -r requirement/requirements.txt
````

<h3>Instructions to score sentences using SEScoreX:</h3>

SEScoreX pretrained weights can be found in google drive: https://drive.google.com/drive/u/2/folders/1TOUXEDZOsjoq_lg616iKUyWJaK9OXhNP


To run SEScoreX for reference based text generation evaluation:


We have SEScore2 that is only pretrained on synthetic data which only supports five languages (version: pretrained)
````
from sescorex import *
scorer = sescorex(version='pretrained', rescale=False)
````


We further fine-tune the pretrained SEScore2 model using WMT17-21 DA data and WMT22 MQM data, which supports up to 100 languages. The model operates in two modes: 'seg' and 'sys'. The 'seg' mode is more effective for ranking pairs of translations, while the 'sys' mode is better suited for ranking translation systems. By default, we select the 'seg' mode.
````
from sescorex import *
scorer = sescorex(version='seg', rescale=False)
````


You can enable the 'rescale' feature to obtain interpretable scores. In this mode, a score of '0' indicates a perfect translation, '-1' corresponds to a translation with one minor error, and '-5' represents a translation with a major error. You can estimate the number of major and minor errors in the translation by counting the multiples of -5 and -1 in the score, respectively. If you prefer the raw output scores, you can disable rescaling by setting rescale=False.
````
from sescorex import *
scorer = sescorex(version='seg', rescale=True)
refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "you went to hotel"]
outs = ["SEScore is a simple effective text evaluation metric for next generation", "you went to zoo"]
scores_ls = scorer.score(refs=refs, outs=outs, batch_size=32)
````

### Supported Languages
Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskrit, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.


### Table: Model Performance Comparison

| Model   | cs-uk | en-cs | en-ja | en-zh | bn-hi | hi-bn | xh-zu* | zu-xh* | en-hr | en-uk | en-af* | en-am* | en-ha* |
|---------|-------|-------|-------|-------|-------|-------|--------|--------|-------|-------|--------|--------|--------|
| XCOMET  | 0.533 | 0.499 | 0.564 | 0.566 | 0.493 | 0.521 | **0.573** | 0.623  | 0.512 | 0.493 | **0.550** | 0.568  | 0.662  |
| COMET22 | **0.550** | **0.522** | **0.580** | **0.586** | 0.503 | **0.528** | 0.564  | 0.657  | **0.551** | **0.540** | 0.548  | 0.570  | **0.693** |
| Ours    | 0.540 | 0.514 | 0.565 | 0.575 | **0.504** | 0.521 | 0.572  | **0.658** | 0.537 | 0.524 | 0.535  | **0.570** | 0.663  |


| Model   | en-ig* | en-rw* | en-lg* | en-ny* | en-om* | en-sn* | en-ss* | en-sw* | en-tn* | en-xh* | en-yo* | en-zu* | en-gu |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|
| XCOMET  | 0.502  | 0.446  | 0.579  | 0.494  | 0.653  | 0.702  | 0.548  | 0.650  | 0.479  | 0.633  | 0.541  | 0.551  | **0.694** |
| COMET22 | **0.539** | 0.456  | 0.582  | **0.535** | 0.672  | 0.807  | 0.580  | **0.679** | **0.605** | 0.692  | 0.575  | 0.589  | 0.596 |
| Ours    | 0.538  | **0.478** | **0.603** | 0.529  | **0.697** | **0.820** | **0.598** | 0.674  | 0.585  | **0.702** | **0.591** | **0.597** | 0.607 |

| Model   | en-hi | en-ml | en-mr | en-ta |
|---------|-------|-------|-------|-------|
| XCOMET  | **0.700** | **0.713** | **0.667** | **0.663** |
| COMET22 | 0.587  | 0.617  | 0.570  | 0.626  |
| Ours    | 0.580  | 0.606  | 0.528  | 0.604  |

**Note:** * indicates African languages.

````
@inproceedings{xu-etal-2023-sescore2,
    title = "{SESCORE}2: Learning Text Generation Evaluation via Synthesizing Realistic Mistakes",
    author = "Xu, Wenda  and
      Qian, Xian  and
      Wang, Mingxuan  and
      Li, Lei  and
      Wang, William Yang",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.283/",
    doi = "10.18653/v1/2023.acl-long.283",
    pages = "5166--5183",
    abstract = "Is it possible to train a general metric for evaluating text generation quality without human-annotated ratings? Existing learned metrics either perform unsatisfactory across text generation tasks or require human ratings for training on specific tasks. In this paper, we propose SEScore2, a self-supervised approach for training a model-based metric for text generation evaluation. The key concept is to synthesize realistic model mistakes by perturbing sentences retrieved from a corpus. We evaluate SEScore2 and previous methods on four text generation tasks across three languages. SEScore2 outperforms all prior unsupervised metrics on four text generation evaluation benchmarks, with an average Kendall improvement of 0.158. Surprisingly, SEScore2 even outperforms the supervised BLEURT and COMET on multiple text generation tasks."
}

@inproceedings{xu-etal-2022-errors,
    title = "Not All Errors are Equal: Learning Text Generation Metrics using Stratified Error Synthesis",
    author = "Xu, Wenda  and
      Tuan, Yi-Lin  and
      Lu, Yujie  and
      Saxon, Michael  and
      Li, Lei  and
      Wang, William Yang",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.489/",
    doi = "10.18653/v1/2022.findings-emnlp.489",
    pages = "6559--6574",
    abstract = "Is it possible to build a general and automatic natural language generation (NLG) evaluation metric? Existing learned metrics either perform unsatisfactorily or are restricted to tasks where large human rating data is already available. We introduce SESCORE, a model-based metric that is highly correlated with human judgements without requiring human annotation, by utilizing a novel, iterative error synthesis and severity scoring pipeline. This pipeline applies a series of plausible errors to raw text and assigns severity labels by simulating human judgements with entailment. We evaluate SESCORE against existing metrics by comparing how their scores correlate with human ratings. SESCORE outperforms all prior unsupervised metrics on multiple diverse NLG tasks including machine translation, image captioning, and WebNLG text generation. For WMT 20/21En-De and Zh-En, SESCORE improve the average Kendall correlation with human judgement from 0.154 to 0.195. SESCORE even achieves comparable performance to the best supervised metric COMET, despite receiving no human annotated training data."
}
````


## Run new_xlm_mbart_data.py for English:
python3 new_xlm_mbart_data.py -num_var 10 -lang en_XX -src case_study_src -ref case_study_ref -save save_file_name

## Run new_xlm_mbart_data.py for German:
python3 new_xlm_mbart_data.py -num_var 10 -lang de_DE -src src_folder -ref ref_folder -save save_file_name
