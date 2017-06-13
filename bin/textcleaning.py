#!/bin/env python
#-*- coding:utf-8 -*-

########################################
#Reference:https://www.kaggle.com/currie32/the-importance-of-cleaning-text
#          https://www.kaggle.com/life2short/data-processing-replace-abbreviation-of-word/notebook
#Author: Yxy
#Date: 2017-05-23
########################################

import re
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
stop_words = stopwords.words("english")

class TextCleaning:
    def __init__(self):
        pass

    @staticmethod
    def clean_text1(text):
        text = str(text).decode('utf-8')
        # 单位处理
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)         # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # 缩略词处理
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # 拼写矫正及词条化
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # 标点符号处理
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)     # 单词与 . 之间插入空格
        text = re.sub(r",", " , ", text)      # 单词与 , 之间插入空格
        text = re.sub(r"\?", " ? ", text)     # 单词与 ? 之间插入空格
        text = re.sub(r"!", " ! ", text)      # 单词与 ! 之间插入空格
        text = re.sub(r"\"", " \" ", text)      # " 左右插入空格
        text = re.sub(r"&", " & ", text)        # & 左右插入空格
        text = re.sub(r"\|", " | ", text)        # | 左右插入空格
        text = re.sub(r";", " ; ", text)        # ; 左右插入空格
        text = re.sub(r"\(", " ( ", text)       # ( 左右插入空格
        text = re.sub(r"\)", " ( ", text)       # ) 左右插入空格

        # 符号替换为单词
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # 去除多余空格
        text = ' '.join(text.split())
        
        return text.encode("utf-8")
        # 单位处理
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)         # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          # e.g. 4k => 4000

        # 缩略词处理
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)

        # 标点符号处理
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\b\.", " . ", text)     # 单词与 . 之间插入空格
        text = re.sub(r"\b,", " , ", text)      # 单词与 , 之间插入空格
        text = re.sub(r"\b\?", " ? ", text)     # 单词与 ? 之间插入空格
        text = re.sub(r"\b!", " ! ", text)      # 单词与 ! 之间插入空格
        text = re.sub(r"\"", " \" ", text)      # " 左右插入空格
        text = re.sub(r"&", " & ", text)        # & 左右插入空格
        text = re.sub(r"|", " | ", text)        # | 左右插入空格
        text = re.sub(r";", " ; ", text)        # ; 左右插入空格
        text = re.sub(r"\(", " ( ", text)       # ( 左右插入空格
        text = re.sub(r"\)", " ( ", text)       # ) 左右插入空格

        # 符号替换为单词
        text = re.sub(r"\&", " and ", text)
        text = re.sub(r"|", " or ", text)
        #text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        #text = re.sub(ur"₹", " rs ", text)      # 测试！

        # 拼写矫正
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)

        # 去除多余空格
        text = ' '.join(text.split())

        return text.encode('utf-8')
    
    @staticmethod
    def clean_text2(text):
        text = str(text).decode('utf-8')
        
        #text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"what's", "what is", text)
        #text = re.sub(r"What's", "what is", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r" usa ", " America ", text)
        text = re.sub(r" USA ", " America ", text)
        text = re.sub(r" u s ", " America ", text)
        text = re.sub(r" uk ", " England ", text)
        text = re.sub(r" UK ", " England ", text)
        text = re.sub(r"india", "India", text)
        text = re.sub(r"switzerland", "Switzerland", text)
        text = re.sub(r"china", "China", text)
        text = re.sub(r"chinese", "Chinese", text) 
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r"quora", "Quora", text)
        text = re.sub(r" dms ", "direct messages ", text)  
        text = re.sub(r"demonitization", "demonetization", text) 
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r"KMs", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text) 
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text) 
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"gps", "GPS", text)
        text = re.sub(r"gst", "GST", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"dna", "DNA", text)
        text = re.sub(r"III", "3", text) 
        text = re.sub(r"the US", "America", text)
        text = re.sub(r"Astrology", "astrology", text)
        text = re.sub(r"Method", "method", text)
        text = re.sub(r"Find", "find", text) 
        text = re.sub(r"banglore", "Banglore", text)
        text = re.sub(r" J K ", " JK ", text)
        
        return text.encode('utf-8')
    
    @staticmethod
    def clean_text3(text):
        text = str(text).decode('utf-8')

        text = re.sub("what's", "what is", text)
        text = re.sub("what're", "what are", text)
        text = re.sub("who's", "who is", text)
        text = re.sub("who're", "who are", text)
        text = re.sub("where's", "where is", text)
        text = re.sub("where're", "where are", text)
        text = re.sub("when's", "when is", text)
        text = re.sub("when're", "when are", text)
        text = re.sub("how's", "how is", text)
        text = re.sub("how're", "how are", text)
    
        text = re.sub("i'm", "i am", text)
        text = re.sub("we're", "we are", text)
        text = re.sub("you're", "you are", text)
        text = re.sub("they're", "they are", text)
        text = re.sub("it's", "it is", text)
        text = re.sub("he's", "he is", text)
        text = re.sub("she's", "she is", text)
        text = re.sub("that's", "that is", text)
        text = re.sub("there's", "there is", text)
        text = re.sub("there're", "there are", text)
    
        text = re.sub("i've", "i have", text)
        text = re.sub("we've", "we have", text)
        text = re.sub("you've", "you have", text)
        text = re.sub("they've", "they have", text)
        text = re.sub("who've", "who have", text)
        text = re.sub("would've", "would have", text)
        text = re.sub("not've", "not have", text)
    
        text = re.sub("i'll", "i will", text)
        text = re.sub("we'll", "we will", text)
        text = re.sub("you'll", "you will", text)
        text = re.sub("he'll", "he will", text)
        text = re.sub("she'll", "she will", text)
        text = re.sub("it'll", "it will", text)
        text = re.sub("they'll", "they will", text)
    
        text = re.sub("isn't", "is not", text)
        text = re.sub("wasn't", "was not", text)
        text = re.sub("aren't", "are not", text)
        text = re.sub("weren't", "were not", text)
        text = re.sub("can't", "can not", text)
        text = re.sub("couldn't", "could not", text)
        text = re.sub("don't", "do not", text)
        text = re.sub("didn't", "did not", text)
        text = re.sub("shouldn't", "should not", text)
        text = re.sub("wouldn't", "would not", text)
        text = re.sub("doesn't", "does not", text)
        text = re.sub("haven't", "have not", text)
        text = re.sub("hasn't", "has not", text)
        text = re.sub("hadn't", "had not", text)
        text = re.sub("won't", "will not", text)
    
        return text.encode('utf-8')

    @staticmethod
    def substitute_thousands(text):
        text = str(text).decode('utf-8')
        matches = re.finditer(r'[0-9]+(?P<thousands>\s{0,2}k\b)', text, flags=re.I)
        result = ''
        len_offset = 0
        for match in matches:
            result += '{}000'.format(text[len(result)-len_offset:match.start('thousands')])
            len_offset += 3 - (match.end('thousands') - match.start('thousands'))
        result += text[len(result)-len_offset:]
        return result.encode('utf-8')
 
    @staticmethod
    def stemming_cleaning(text):
        sentences = nltk.sent_tokenize(str(text).decode('utf-8'))
        words = []
        for sent in sentences:
            words += nltk.word_tokenize(sent) 
        words = [stemmer.stem(word) for word in words]
        
        return " ".join(words).encode('utf-8')
   
    @staticmethod
    def stopword_cleaning(text):
        sentences = nltk.sent_tokenize(str(text).decode('utf-8'))
        #sentences = nltk.sent_tokenize(str(text))
        words = []
        for sent in sentences:
            words += nltk.word_tokenize(sent) 
        words = [word for word in words if word not in stop_words]
        
        return " ".join(words).encode('utf-8')
    
    @staticmethod
    def punctuation_cleaning(text):
        text = str(text).decode('utf-8')
        # Remove punctuation from text
        text = ''.join([c for c in text if c not in punctuation])
        
        return text.encode('utf-8')

    @staticmethod
    def text_cleaning(text, lower=False, stem=False, stopword=False, punctutation=False):
        if lower:
            text = str(text).decode('utf-8').lower().encode('utf-8')
        text = TextCleaning.clean_text1(text)
        text = TextCleaning.clean_text2(text)
        #print "Yes!"
        #text = TextCleaning.clean_text3(text)
        text = TextCleaning.substitute_thousands(text)
        if stem:
            text = TextCleaning.stemming_cleaning(text)
        if stopword:
            text = TextCleaning.stopword_cleaning(text)
        if punctutation:
            text = TextCleaning.punctuation_cleaning(text)
        text = ' '.join(text.decode('utf-8').split()).encode('utf-8')
        return text

#print TextCleaning.text_cleaning("I'm a         good boy!")
#print TextCleaning.text_cleaning("What is the best/most memorable thing you've ever eaten and why?", stem=True, stopword=True, punctutation=True)
#print TextCleaning.text_cleaning("When do you use シ instead of し?", stem=True, stopword=True, punctutation=True)
#"does, do"
