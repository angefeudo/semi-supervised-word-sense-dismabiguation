__author__ = "Angelo Fiestada"
__email__ = "angelofiestada@gmail.com"

import re
import sys
import nltk
import time
import operator
import py_compile
import numpy as np
import itertools
from stat_parser import Parser
from collections import Counter
from itertools import takewhile
from dictionary import Dictionary
from nltk.corpus import stopwords
from PyQt5.QtCore import pyqtSlot, Qt
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag, FreqDist
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QIcon, QFont
from sklearn.feature_extraction.text import TfidfVectorizer
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QSizePolicy, QPlainTextEdit	 

parser = Parser()
vectorizer = TfidfVectorizer()
d = Dictionary()
d.load("DICTIONARYFIN.txt")
remv = ['1', 'n', 'v', 'adj', 'article', 'interj', 'adv']
with open ("english_words.txt") as word_file:
	ew = set(word.strip().lower() for word in word_file)

with open ("tagalog_words.txt") as word_file:
	tw = set(word.strip().lower() for word in word_file)


def get_def(word):
	wd = d.has_word(word)
	if any(c.isdigit() for c in wd):
		return re.split(' 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |18 |19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 29 | 30 | 31 | 33 | 34 | 35 ',wd)
	else:
		return wd



def process_inpt(sen):
	dic = {}
	wlist = []
	for w in sen.split():
		wd = get_def(w)
		dic.update({"w":w,"e_sense":list(map(split_eng, wd)), "t_sense":list(map(split_tag, wd))})
		wlist.append(dic.copy())
	return wlist

def get_sim_score(dictcontent, inpt):
    superfinalscore = []
    #print("dictcontent in get_sim_score "+str(dictcontent))

    fininptsyn = inpt
    for findictsyn in dictcontent:
        if findictsyn:
            fscore = []
            for findictsy in findictsyn:
                #print("findictsy  ->>>>>>>>>"+str(findictsy))
                if fininptsyn and findictsy:
                    pairlist = [(inptsyn, dictsyn) for inptsyn in fininptsyn for dictsyn in findictsy]
                    score = [pair[0].path_similarity(pair[1]) for pair in pairlist]
                    sc= sum([x for x in score if x is not None])/len(pairlist)
                    if sc:
                    #print(sc)
                        fscore.append(sc)
            if fscore:
                finalscore = sum([x for x in fscore if x is not None])/len(findictsyn)
                superfinalscore.append(finalscore)
    return superfinalscore

def split_eng(dftn):
    global e
    e = ew
    dftn2 = dftn.split(":")
    mslist= []
    for df in dftn2:
        a = " ".join(w for w in nltk.wordpunct_tokenize(df)
    		if w.lower() in e or  not w.isalpha())
        inpt = ' '.join(i for i  in re.sub(r'[^\w]', ' ', a).split() if i not in remv)
        #print("inpt"+str(inpt))
        try:
            posinstr = ' '.join(str(parser.parse(str(inpt))).split())
            #print("*****PARSED***"+str(posinstr))
            parsedcon = list(np_vp_extractor(posinstr))
            #print("parsedcon "+str(parsedcon))
            wordposin = [word for word in parsedcon if len(re.findall(r'\w+', word)) == 2]
            #print("wordposin "+str(wordposin))
            #print("posinstr "+str(posinstr))
            cleanin =[(w.split()[0], w.split()[1]) for w in wordposin if w.split()[1] not in stopwords.words("english")]
            #print("cleanin "+str(cleanin))
            inptsyn =  [tagged_to_synset(*tagged_word) for tagged_word in cleanin]
            #print("inptsyn  "+str(inptsyn))
            m =  [ss for ss in inptsyn if ss]
            mslist.append(m)
            return mslist
        except:
            return None


def split_tag(dftn):
    global e
    e = tw
    a = " ".join(w for w in nltk.wordpunct_tokenize(dftn)
		if w.lower() not in e or  not w.isalpha())

    c = ' '.join(i for i  in re.sub(r'[^\w]', ' ', a).split() if i not in remv)
    #print(c)
    return c


def np_vp_extractor(posstring):
    stack = []
    for i, c in enumerate(posstring):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            #if len(stack) == 1:
            text = yield(str(posstring[start + 1: i]))

def tagged_to_synset(tag, word):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def penn_to_wn(tag):
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    return None


def disambiguate(sentence):
    l = []
    w = ' '.join([i for i in sentence.split() 
        if i not in stopwords.words("english")])
    print("w "+ str(w))
    word_dictionary = process_inpt(w)
    for wr in word_dictionary:
        print("wr[e_sense] "+str(wr["e_sense"]))
        print()
        print("wr[t_sense] "+str(wr["t_sense"]))
	#parsed = preprocess(sentence)
    slen = len(sentence)


    posinstr = ' '.join(str(parser.parse(str(sentence))).split())
    print("INPUT"+str(posinstr))
    parsedcon = list(np_vp_extractor(posinstr))
    wordposin = [word for word in parsedcon if len(re.findall(r'\w+', word)) <= 3]
    cleanin =[(w.split()[0], w.split()[1]) for w in wordposin]
    inptsyn =  [tagged_to_synset(*tagged_word) for tagged_word in cleanin]
    parsed = [ss for ss in inptsyn if ss]


    k = [w["t_sense"][np.argmax(get_sim_score(w["e_sense"], parsed))].split() for w in word_dictionary]

    return k
'''
def output():

    for s in k:
        print("k   "+str(s))
    merged = list(itertools.chain(*k))
    print(merged)
    return " ".join(Counter(w["t_sense"][np.argmax(get_sim_score(w["e_sense"], parsed))].split()).most_common()[0][0] for w in word_dictionary)
    return " ".join(Counter(x).most_common()[0][0] for x in k)

def output_asso_words():
    merged = list(itertools.chain(*k))
    return  " ".join(x for x in merged)
'''


appStyle = """
QMainWindow{
background-image: url(24.png);
	
}"""
class App(QtWidgets.QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = 'Translation Selection'
        self.left = 100
        self.top = 100
        self.width = 840
        self.height = 580
        self.initUI()
        self.setStyleSheet(appStyle)	

 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.font4 = QFont()
        self.font4.setFamily("Arial")
        self.font4.setPointSize(11)



        self.font5 = QFont()
        self.font5.setFamily("Cooper Black")
        self.font5.setPointSize(11)


        self.label = QLabel()
        self.label.setText("Translation Selection")
        self.label.setAlignment(Qt.AlignCenter)
        self.font = QFont()
        self.font2 = QFont()
        self.font2.setFamily("Cooper Black")
        self.font2.setPointSize(40)
        self.font.setFamily("Arial")
        self.font.setPointSize(18)
        # Create textbox for input
        self.outbox = QLineEdit(self)
        self.outbox.move(70, 300)
        self.outbox.resize(700,60)
        self.outbox.setFont(self.font)


        self.words = QLineEdit(self)
        self.words.move(70, 400)
        self.words.resize(700,60)
        self.words.setReadOnly(True)
        self.words.setFont(self.font4)




        self.font3 = QFont()
        self.font3.setFamily("Cooper Black")
        self.font3.setPointSize(20)

        self.label = QLabel("Translation Selection", self)
        self.label.setStyleSheet('QLabel {color: #112233;}')
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.move(100,50)
        self.label.setFont(self.font2)
        self.label.setGeometry(QtCore.QRect(-100, -450, 1000, 1000))

        self.label2 = QLabel("Input Sentence", self)
        self.label2.setStyleSheet('QLabel {color: #112233;}')
        self.label2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label2.setAlignment(Qt.AlignCenter)
        #self.label2.move(800,800)
        self.label2.setFont(self.font3)
        self.label2.setGeometry(QtCore.QRect(-325, -320, 1000, 1000))



        self.label3 = QLabel("Translation", self)
        self.label3.setStyleSheet('QLabel {color: #112233;}')
        self.label3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label3.setAlignment(Qt.AlignCenter)
        #self.label2.move(800,800)
        self.label3.setFont(self.font3)
        self.label3.setGeometry(QtCore.QRect(-345, -220, 1000, 1000))


        self.label4 = QLabel("Associated Words", self)
        self.label4.setStyleSheet('QLabel {color: #112233;}')
        self.label4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label4.setAlignment(Qt.AlignCenter)
        #self.label2.move(800,800)
        self.label4.setFont(self.font3)
        self.label4.setGeometry(QtCore.QRect(-310, -120, 1000, 1000))


        self.w = QWidget()
        self.w.resize(200, 100)
        self.inputTexbox = QPlainTextEdit(self.w)
        self.inputTexbox.move(70,500)



        # create textbox for output
        self.inbox = QLineEdit(self)
        self.inbox.move(70, 200)
        self.inbox.resize(700, 60)
        self.inbox.setFont(self.font)
 
        self.button = QPushButton('Translate', self)
        self.button.move(350,500)
        self.button.setFont(self.font5)

        self.button.clicked.connect(self.on_click)
        self.show()


    @pyqtSlot()
    def on_click(self):
        textboxValue = self.inbox.text()
        self.outbox.setText(" ".join(Counter(x).most_common()[0][0] for x in disambiguate(textboxValue)))
        self.words.setText(" ".join(x for x in list(itertools.chain(*disambiguate(textboxValue)))))
if __name__ == '__main__':

	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())	