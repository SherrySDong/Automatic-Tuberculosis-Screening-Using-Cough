TABLE = open('preparegs.txt', 'w')
meta = open('./CODA_TB_Solicited_Meta_Info.csv', 'r')
meta.readline()
count = {}
tb_status = {}
for line in meta:
    line = line.strip()
    table = line.split(',')
    if table[0] in count:
        count[table[0]] = count[table[0]] + 1
    else:
        count[table[0]] = 1
meta.close()

FILE = open('./Clinical.tsv', 'r')
FILE.readline()
for line in FILE:
    line = line.strip()
    table = line.split('\t')
    tb_status[table[0]] = table[-1]
FILE.close()


meta = open('./CODA_TB_Solicited_Meta_Info.csv', 'r')
meta.readline()
for line in meta:
    line = line.strip()
    table = line.split(',')
    TABLE.write(table[1])
    TABLE.write('\t')
    TABLE.write(tb_status[table[0]])
    TABLE.write('\t')
    TABLE.write(str(count[table[0]]))
    TABLE.write('\t')
    TABLE.write(table[0])
    TABLE.write('\n')
