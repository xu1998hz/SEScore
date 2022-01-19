src_file = open('train/zhen_train/src/gpu7/19_train.src.zhen', 'r')
ref_file = open('train/zhen_train/ref/gpu7/19_train.ref.zhen', 'r')
new_src_file = open('train/zhen_train/ref/gpu7/new_19_train.src.zhen', 'w')
new_ref_file = open('train/zhen_train/ref/gpu7/new_19_train.ref.zhen', 'w')

count = 0
for src_line, ref_line in zip(src_file, ref_file):
    if len(ref_line.split()) > 1:
        new_src_file.write(src_line)
        new_ref_file.write(ref_line)
        count+=1
print(count)
