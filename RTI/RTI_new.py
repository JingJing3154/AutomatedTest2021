# -*- coding = utf-8 -*
# @Time : 2021/11/01 19:10
# @Author : JingJing
# @File : new.py
# @Software : PyCharm
from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
import numpy as np
import string
import re
from collections import Counter
import os, requests, uuid, json
from translate import Translator

#调用bing翻译
def BingTranslate(api_key, text, language_from, language_to):
    #这里采用
    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&language=' + language_from + '&to=' + language_to
    constructed_url = base_url + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    if type(text) is str:
        text = [text]

    body = [{'text': x} for x in text]
    # you can pass more than one object in body.

    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    return [i["translations"][0]["text"] for i in response]

#计算在句子中出现的单词的频率
# np_super_target: 所包含的短语或句子
# np_sub_target: the RTI
# distance: 距离阈值：计算方法由pdf给出
# return: 关于对句子翻译有效性的判断
def CheckInvariantDict(np_super_target, np_sub_target, distance):
    violation = False
    currentdistance = 0
    #计算每个汉字的数量
    np_super_targetD = dict(Counter(np_super_target))
    np_sub_targetD = dict(Counter(np_sub_target))

    #计算每个每个元素的数量
    # np_super_targetD = dict(Counter(np_super_target.split()))
    # np_sub_targetD = dict(Counter(np_sub_target.split()))

    #对于汉语中的语句结尾
    punctsL = ['“', '”', '，', '·', ' ', ',', '的', '了', "'", '"']
    #计算说明：其中对于两个在对比的元素来说，相较于前一个元素集，若出现了相同的原宿则认为距离是不变的，若不存在则认为存在距离为1
    #Bow词袋距离的计算方法
    for pun in punctsL:
        np_sub_targetD.pop(pun, None)

    for key in np_sub_targetD:
        if key not in np_super_targetD:
            currentdistance += np_sub_targetD[key]

        else:
            value_difference = np_sub_targetD[key] - np_super_targetD[key]
            if value_difference > 0:
                currentdistance += value_difference
    #最终判断，比较两者之间的距离用来判断两个词袋之间的差异
    if currentdistance > distance:
        violation = True

    return violation

# 一种考虑在RTI翻译中只出现单词/字符的替代方法
# np_super_target: the containing phrase/sentence
# np_sub_target: the RTI
# distance: 距离阈值
# return: 对于距离阈值的判断
def CheckInvariantSet(np_super_target, np_sub_target, distance):
    violation = False

    # count the number of every Chinese char
    #此处创建无序不重复元素集
    np_super_targetS = set(list(np_super_target))
    np_sub_targetS = set(list(np_sub_target))

    # count the number of every words
    # np_super_targetD = dict(Counter(np_super_target.split()))
    # np_sub_targetD = dict(Counter(np_sub_target.split()))

    # additional "stop" chars in Chinese
    stopS = {'“', '”', '，', '·', ' ', ',', '的', '了', "'", '"'}

    diffS = np_sub_targetS - np_super_targetS

    diffS = diffS - stopS

    if len(diffS) > distance:
        violation = True

    return violation

# 找到潜在RTIs并存储在“invariantsD”中
# super_node:选区解析树中的非终端节点
# super_str:非终端节点对应的短语
# invariantsD: 一个RTIs的字典，每个键是一个包含的RTI，每个值是该键包含的RTIs的列表
# stopwordsS: 一组停顿词
# num_word_th: RTI的最大长度
#这里用到了stanford的词法分析工具，其中叶节点由名词构成
def FindInvariant(super_node, super_str, invariantsD, stopwordsS, num_word_th):
    for node in super_node:

        # 检查是否为叶节点
        if isinstance(node, str):
            continue

        label = node.label()
        if label == 'NP':
            if len(node.leaves()) == 1:
                continue

            # 获取非终端节点对应的短语
            node_str = ' '.join(node.leaves()).replace(" 's", "'s").replace("-LRB-", "(").replace("-RRB-", ")").strip()
            node_strL = node_str.split()

            num_word_no_stop = len([i for i in node_strL if i not in stopwordsS])

            # if the RTI is not too long and not too short
            if len(node.leaves()) < num_word_th and num_word_no_stop > 2:

                if node_str.endswith("'"):
                    node_str = node_str[:-1].strip()

                if super_str not in invariantsD:
                    invariantsD[super_str] = set([node_str])
                else:
                    invariantsD[super_str].add(node_str)

            # 递归搜索包含的RTIs/句子
            FindInvariant(node, node_str, invariantsD, stopwordsS, num_word_th)

        FindInvariant(node, super_str, invariantsD, stopwordsS, num_word_th)

# Purity implementation
# output_filename: output file name, the default path is "filename_bugs_distance.txt"
# distance_threshold: the distance threshold
# rtiD: the RTIs found
def RTI(output_filename, distance_threshold, nmtsoftware, rtiD, source_lang, target_lang):
    suspicious_issuesL = []
    non_suspicious_issuesL = []
    numberOfChar = 0
#结果输出路径
    output_file = output_filename + '_bugs_' + str(distance_threshold) + '.txt'
    write_output = open(output_file, 'w')

    # optional, also output the non_suspicious issues
    # non_suspicious_file = output_filename+'_corrects_'+str(distance_threshold)+'.txt'
    # write_correct = open(non_suspicious_file, 'w')

    translated_count = 0
    # 在字典中检查每个包含RTIs/句子及其所包含的RTIs
    for invIdx, np_super in enumerate(rtiD):

        np_subS = rtiD[np_super]

        # For the super phrase

        # 如果已经翻译，直接获取已翻译好的部分
        if np_super in cached_translationsD:
            np_super_target = cached_translationsD[np_super]

        # If not translated, use API
        else:
            numberOfChar += len(np_super)
            if nmtsoftware == 'google':
                # Google translate
                translation = google_translate(np_super)
                np_super_target = translation.replace("&#39;", "'")
            else:
                # Bing translate
                np_super_target = BingTranslate(apikey, np_super, source_lang, target_lang)[0]

            cached_translationsD[np_super] = np_super_target
            translated_count += 1

        # 过滤一些奇怪的翻译并返回
        if re.search('[a-z]', np_super_target):
            # print ('super illegal:', np_super_target)
            continue

        # For the subset phrase
        for np_sub in np_subS:
            if np_sub in cached_translationsD:
                np_sub_target = cached_translationsD[np_sub]
            else:
                numberOfChar += len(np_sub)
                if nmtsoftware == 'google':
                    # Google translate
                    translation = google_translate(np_sub)
                    np_sub_target = translation.replace("&#39;", "'")
                else:
                    # Bing translate
                    np_sub_target = BingTranslate(apikey, np_sub, source_lang, target_lang)[0]

                cached_translationsD[np_sub] = np_sub_target
                translated_count += 1

            # filter some strange translations returned
            if re.search('[a-z]', np_sub_target):
                # print ('sub illegal:', np_sub_target)
                continue

            # 如果改变了句子中原有的固定短语，则作为bug报告
            # violation = CheckInvariantSet(np_super_target, np_sub_target, distance_threshold)
            violation = CheckInvariantDict(np_super_target, np_sub_target, distance_threshold)

            if violation:
                suspicious_issuesL.append((np_super, np_super_target, np_sub, np_sub_target))
    # else:
    # 	non_suspicious_issuesL.append((np_super, np_super_target, np_sub, np_sub_target))

    # Print all the issues
    for idx, issue in enumerate(suspicious_issuesL):
        write_output.write('Issue: ' + str(idx) + '\n')
        write_output.write(issue[0] + '\n')
        write_output.write(issue[1] + '\n')
        write_output.write(issue[2] + '\n')
        write_output.write(issue[3] + '\n')

    print('There are 2 API calls.')

    # for idx, issue in enumerate(non_suspicious_issuesL):
    # 	write_correct.write('Issue: ' + str(idx) + '\n')
    # 	write_correct.write(issue[0] + '\n')
    # 	write_correct.write(issue[1] + '\n')
    # 	write_correct.write(issue[2] + '\n')
    # 	write_correct.write(issue[3] + '\n')

    write_output.close()
    # write_correct.close()

    return numberOfChar

#################################################################
########################Main Code################################
#################################################################
if __name__ == "__main__":
    #默认阈值为0
    #print("start program")
    distance_threshold = 0
    dataset = 'business'
    #优先使用google翻译库
    software = 'google'
    num_word_th = 10
    numberOfChar = 0

    # initialize a constituency parser
    #选用斯坦福文本分析工具NLP,使用时要注意端口的开放
    eng_parser = CoreNLPParser('http://localhost:9001')

    # initialize the Google translate client
    def google_translate(input):
        #translator = Translator(service_urls=[
        #    'translate.google.com',
        #])
        translator = Translator(from_lang="english", to_lang="chinese")
        res = []
        for origin in input:
            trans = translator.translate(origin)
            res.append(trans)
        return res



    # input your key for Bing Microsoft translator
    #因安全原因选择隐藏
    apikey = '*******************************'

    # set original sentence file path
    input_file = './data/' + dataset

    # for google en->zh
    #此处默认为翻译从英文到中文
    source_lang = 'en'
    target_lang = 'zh-CN'

    # for bing en-zh
    # source_lang = 'en'
    # target_lang = 'zh-Hans'

    output_filename = './data/' + dataset + '_' + software
    #print(output_filename)
    # initialize stop words in the source language
    stopwordsS = set(stopwords.words('english'))
    #注：stopwords通过nltk重新生成

    # get original sentences from file
    ori_source_sents = []
    with open(input_file) as file:
        for line in file:
            ori_source_sents.append(line.strip())
    #print(ori_source_sents)
    # 一个RTIs的字典，每个键是一个包含的RTI，每个值是该键包含的RTIs的列表
    np_invariantsD = dict()

    # 解析原句,词法分析树
    ori_source_trees = [i for (i,) in eng_parser.raw_parse_sents(ori_source_sents, properties={'ssplit.eolonly': 'true'})]

    # find RTIs
    for t, super_str in zip(ori_source_trees, ori_source_sents):
        FindInvariant(t, super_str, np_invariantsD, stopwordsS, num_word_th)

    print('\n invariants constructed\nThere are', len(np_invariantsD), 'invariants. Filtering')
    ##print(ori_source_sents)
    # 由于np有时可能与原句几乎相同(即，只有标点不同)，我们在这里过滤这些重复对。
    chartosent = dict()
    for sent in ori_source_sents:
        sent_no_pun = ''.join(sent.translate(str.maketrans('', '', string.punctuation)).strip().split())
        chartosent[sent_no_pun] = sent
    #print("已过滤重复文件")
    # rtiD是RTI对的字典
    rtiD = dict()
    for super_str in np_invariantsD:
        super_str_no_pun = ''.join(super_str.translate(str.maketrans('', '', string.punctuation)).strip().split())

        if super_str_no_pun in chartosent:
            sent = chartosent[super_str_no_pun]
            if sent in rtiD and len(np_invariantsD[super_str]) < len(rtiD[sent]):
                continue
            rtiD[sent] = np_invariantsD[super_str]
        else:
            rtiD[super_str] = np_invariantsD[super_str]

    print('\n invariants filtered\nThere are', len(rtiD), 'invariants.')
    # 原始的句子翻译
    ori_target_sents = []
    for ori_source_sent in ori_source_sents:
        if software == 'google':
            print("google")
            # Google translate
            translation = google_translate(ori_source_sent)
            ori_target_sent = str(translation).replace("&#39;", "'")
        else:
            print("bing")
            # Bing translate
            ori_target_sent = BingTranslate(apikey, ori_source_sent, source_lang, target_lang)[0]
            print(ori_target_sent)

        numberOfChar += len(ori_source_sent)
        ori_target_sents.append(ori_target_sent)

    # 字典要记住所有翻译过的句子，将被重复使用进行不变性检查
    cached_translationsD = dict()
    for source_sent, target_sent in zip(ori_source_sents, ori_target_sents):
        cached_translationsD[source_sent] = target_sent

    # 检测翻译
    tempChar = RTI(output_filename=output_filename, distance_threshold=distance_threshold, nmtsoftware=software, rtiD=rtiD,
                   source_lang=source_lang, target_lang=target_lang)
    numberOfChar += tempChar

    print('Number of characters translated:', tempChar)