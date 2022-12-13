# SEScore
This repo contains all the codes for SEScore implementation. SEScore is a reference-based text-generation evaluation metric that requires no pre-human-annotated error data, described in our paper [Not All Errors are Equal: Learning Text Generation Metrics using Stratified Error Synthesis.]("https://arxiv.org/abs/2210.05035") from EMNLP 2022.

How to run our code?
We hosted our SEScore metric and running instructions on HuggingFace: https://huggingface.co/spaces/xu1998hz/sescore

Run new_xlm_mbart_data.py for English:
python3 new_xlm_mbart_data.py -num_var 10 -lang en_XX -src case_study_src -ref case_study_ref -save save_file_name

Run new_xlm_mbart_data.py for German:
python3 new_xlm_mbart_data.py -num_var 10 -lang de_DE -src src_folder -ref ref_folder -save save_file_name
