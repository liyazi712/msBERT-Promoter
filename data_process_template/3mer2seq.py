import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# 读取tsv文件
df = pd.read_csv('./sample_data/ft/promoter2non_promoter/3/dev.tsv', sep='\t', header=None)
sequences = df[0].tolist()

# 删除每三个序列之间的空格
for i in range(len(sequences)):
    sequence = sequences[i]
    new_sequence = ''.join(sequence.split())
    result = new_sequence[:3]  # 提取第1-3个元素
    for k in range(2, 79):
        index = 3 * k - 1
        if index < len(new_sequence):
            result += new_sequence[index]
    result += new_sequence[-1]
    sequences[i] = result

# 创建一个空的SeqRecord列表
records = []
for i in range(len(sequences)):
    seq = Seq(sequences[i])
    record = SeqRecord(seq, id=f"sequence_{i}", description="")
    records.append(record)

# 保存SeqRecord列表为.fasta格式的文件
output_file = "promoter2non_promoter.fasta"
SeqIO.write(records, output_file, "fasta")

print(f"处理后的序列已保存到 {output_file}")


