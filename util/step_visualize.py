import csv
import re
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM

tsvFile = open('jan_10_zhen_news_comp_zhen_num_0.5_del_1.5_mask_1.5_xlm_mbart.tsv', 'r')
tsvReader = csv.reader(tsvFile, delimiter="\t")

xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
mbart_tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang='en_XX')

score_step_file = open('score_step.txt', 'w')
total = 0
mbart_addition = 0
for row in tsvFile:
    cur_row = row.split('\t')
    original = cur_row[0]
    seg_ls = []
    for seg_row in cur_row:
        seg_ls.append(seg_row.split(' [Score:')[0])
    total += 1
    # print("Extracted all the sentences!")
    if len(cur_row) > 1:
        score_step_file.write(f"Original sentence: { original}\n\n")
        # index is always one before current
        for index, cur_row_ele in enumerate(cur_row[1:]):
            err_type = cur_row_ele.split('Info: [')[1].split(',')[0][1:-1]
            if err_type[:5] == 'MBart':
                if err_type.split()[1] == "Addition":
                    mbart_addition += 1
                    slice_index = int(cur_row_ele.split('Info: [')[1].split(', ')[1].split(']')[0])+1
                    left_side = ''.join(mbart_tokenizer.tokenize(seg_ls[index])[:slice_index])
                    right_side = ''.join(mbart_tokenizer.tokenize(seg_ls[index])[slice_index:])
                    prior = re.sub('▁', ' ', left_side)[1:] + " <MBart Inserts MASK> " + re.sub('▁', ' ', right_side)
                    mbart_addition+=1
                else:
                    start_index = int(cur_row_ele.split("['MBart Replace', ")[1].split(',')[0])+1
                    span = int(cur_row_ele.split("['MBart Replace', ")[1].split(', ')[1].split(']')[0])
                    left_side = ''.join(mbart_tokenizer.tokenize(seg_ls[index])[:start_index])
                    span_region = mbart_tokenizer.tokenize(seg_ls[index])[start_index:start_index+span]
                    right_side = ''.join(mbart_tokenizer.tokenize(seg_ls[index])[start_index+span:])
                    prior = re.sub('▁', ' ', left_side)[1:] + f" <MBart Replace {span_region}> " + re.sub('▁', ' ', right_side)
                    # print(seg_ls[index+1])
                    # print('-----------------')
            elif err_type[:3] == 'XLM':
                if err_type.split()[1] == "Addition":
                    slice_index = int(cur_row_ele.split('Info: [')[1].split(', ')[1].split(']')[0])+1
                    left_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[:slice_index])
                    right_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[slice_index:])
                    prior = re.sub('▁', ' ', left_side)[1:] + " <XLM Inserts MASK> " + re.sub('▁', ' ', right_side)

                    # print(prior)
                    # print(seg_ls[index+1])
                    # print('-----------------------------')
                else:
                    start_index = int(cur_row_ele.split("['XLM Replace', ")[1].split(',')[0])+1
                    span = int(cur_row_ele.split("['XLM Replace', ")[1].split(', ')[1].split(']')[0])
                    #print(cur_row_ele)
                    left_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[:start_index])
                    span_region = xlm_tokenizer.tokenize(seg_ls[index])[start_index:start_index+span]
                    right_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[start_index+span:])
                    prior = re.sub('▁', ' ', left_side)[1:] + f" <XLM Replace {span_region}> " + re.sub('▁', ' ', right_side)
                    # print(cur_row[index+1].split('[Score: ')[1].split(', Info:')[0])
                    # print(prior)
                    # print(seg_ls[index+1])
                    # print('-----------------------------')
            elif err_type[:6] == 'Delete':
                start_index = int(cur_row_ele.split("['Delete', ")[1].split(',')[0])+1
                span = int(cur_row_ele.split("['Delete', ")[1].split(', ')[1].split(']')[0])
                left_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[:start_index])
                span_region = xlm_tokenizer.tokenize(seg_ls[index])[start_index:start_index+span]
                right_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[start_index+span:])
                prior = re.sub('▁', ' ', left_side)[1:] + f" <Delete {span_region}> " + re.sub('▁', ' ', right_side)
                # print(cur_row[index+1].split('[Score: ')[1].split(', Info:')[0])
                # print(prior)
                # print(seg_ls[index+1])
                # print('-----------------------------')
            else:
                start_index = int(cur_row_ele.split("['Switch', ")[1].split(',')[0])+1
                end_index = int(cur_row_ele.split("['Switch', ")[1].split(', ')[1].split(']')[0])
                left_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[:start_index])
                l_token = f" <Switch: {xlm_tokenizer.tokenize(seg_ls[index])[start_index]}> "
                middle = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[start_index+1:end_index])
                r_token = f" <Switch: {xlm_tokenizer.tokenize(seg_ls[index])[end_index]}> "
                right_side = ''.join(xlm_tokenizer.tokenize(seg_ls[index])[end_index+1:])

                prior = re.sub('▁', ' ', left_side)[1:]+re.sub('▁', ' ', l_token)+re.sub('▁', ' ', middle)+re.sub('▁', ' ', r_token)+re.sub('▁', ' ', right_side)

                # print(cur_row[index+1].split('[Score: ')[1].split(', Info:')[0])
                # print(prior)
                # print(seg_ls[index+1])
                # print('-----------------------------')

            score_step_file.write(f"Score: {cur_row[index+1].split('[Score: ')[1].split(', Info:')[0]}\n")
            score_step_file.write('Prior Step Sentence: \n')
            score_step_file.write(prior+'\n')
            score_step_file.write('Modified Sentence based on noise: \n')
            score_step_file.write(seg_ls[index+1]+'\n\n')

        score_step_file.write('-----------------------------Finish One Noise sentence generation through above steps--------------------------------\n\n')

print(mbart_addition)
print(total)
