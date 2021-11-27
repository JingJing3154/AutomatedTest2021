## Read Me

该工具使用到了词法分析工具，我们在这里使用stanford-corenlp。

下载链接：https://box.nju.edu.cn/f/1e629b71c80c4c3cac58/

在运行程序之前需要与NLP工具达成连接：

Run `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -ServerProperties props.properties -preload tokenize,ssplit,pos,lemma,ner,parse -status_port 9000 -port 9000`

或是点击工具文件夹中 start.bat 脚本

该工具实质上是翻译工具，翻译的源文件在./data路径下

随后开始运行程序，程序会给出此次翻译的正确性。

运行视频：https://box.nju.edu.cn/f/655ff1bcb721482f81e1/