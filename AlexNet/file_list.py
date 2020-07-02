def file_class_list(filename):
    # 将文件名和文件标签读出字典， 返回字典
    fr = open(filename)
    train_class = {}  # prepare dictionary to return
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        listFromLine = listFromLine[0:4]
        train_class[listFromLine[0]] = listFromLine[1] # [‘T1000001.wav’: genuine]
    fr.close()
    return train_class


def write_jpg_list(filename, newfile):
    # 输入wav标签文件，写成jpg文件标签,存入newfile
    # 写入格式：‘T1000001.jpg 1’
    fr = open(filename, 'r')
    fw = open(newfile, 'w')
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        rename = listFromLine[0].split('.')
        rename = rename[0] + '.jpg '
        re_class = '0' if listFromLine[1] == 'genuine' else '1'
        newrecording = rename + re_class + '\n'
        fw.write(newrecording)
    fr.close()
    fw.close()


def readfile(filename):
    fr = open(filename)
    for line in fr.readlines():
        print(line)
    fr.close()


if __name__ == '__main__':  # 文件作为脚本直接执行，而 import 到其他脚本中是不会被执行的。
    write_jpg_list('../ASVspoof2017/protocol/ASVspoof2017_V2_train.trn.txt', '../ASVspoof2017/train_jpg.txt')
    write_jpg_list('../ASVspoof2017/protocol/ASVspoof2017_V2_dev.trl.txt', '../ASVspoof2017/dev_jpg.txt')
    write_jpg_list('../ASVspoof2017/protocol/ASVspoof2017_V2_eval.trl.txt', '../ASVspoof2017/eval_jpg.txt')
    readfile('../ASVspoof2017/dev_jpg.txt')
# train_class_dict = file_class_list("D:/b 所有课程/1-Speech_Recognition_code/ASVspoof2017/train_test.txt")
# print(train_class_dict['T_1000001.wav'])

