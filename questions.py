import nltk
import sys
import os
import string
import numpy

FILE_MATCHES = 1
SENTENCE_MATCHES = 1
QUESTION_WORDS = ["how", "what", "where", "when", "why", "who", "which"]

def main():
    print("Technoplus question answering AI")
    print("(C) 2020 Technoplus inc, code by Bala Venkataraman")
    filen = input("Filename:")
    # Calculate IDF values across files
    files = load_files(filen)
    print("tokenizing words")
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)
    print("tokenization complete")
    # Prompt user for query
    query = set(tokenize(input("Question: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)
    exit = input("Do you want to exit ? ")
    exit1 = exit.lower()
    while exit1 == "no":
        query = set(tokenize(input("Quesion: ")))
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)
        exit = input("Do you want to exit ? ")
        exit1 = exit.lower()

def load_files(directory):
    #returns a dictionary mapping the filename of each text file inside that directory to the file's contents as a string.
    a = os.listdir(directory)
    filedict = {}
    for i in a:
        print("loading file: ",end="")
        print(i)
        f = open(os.path.join(directory, i),"r",encoding='utf-8')
        st = ""
        for x in f:
            st +=  x
        filedict[i] = st
        f.close()
    print("loading complete")    
    return filedict
def tokenize(document):
    #returns a list of all of the words in that document, in order. Processes document by coverting all words to lowercase, and removes any punctuation or English stopwords.

    tokens = nltk.word_tokenize(document)
    ctokens = [] 
    for i in tokens:
        if not i in string.punctuation and not i in nltk.corpus.stopwords.words("english"):
            ctokens.append(i.lower())
    return ctokens        
def compute_idfs(documents):
    #return a dictionary that maps words to their IDF values. Any word that appears in at least one of the documents should be in the resulting dictionary.
    words = []
    scores = []
    docs = list(documents.keys())
    for i in docs:
        for j in documents[i]:
            if not j in words:
                words.append(j)
    for i in words:
        ndw = 0
        for j in docs:
            if i in documents[j]:
                ndw += 1
        scores.append(numpy.log(ndw/len(docs)))
    wscores = {}
    for i in range(len(words)):
        wscores[words[i]] = scores[i]
    return wscores    

                            

def top_files(query, dictfiles, idfs, n):
    #return the n most relavent files
    files = list(dictfiles.keys())
    filescores = []
    for i in files:
        score = 0
        for j in dictfiles[i]:
            if j in query:
                idf = idfs[j]
                score += idf
        filescores.append(score)
    
    for i in range(len(filescores)):
        for j in range(len(filescores) - 1):
            if filescores[j] > filescores[i]:
                temp = filescores[i]
                filescores[i] = filescores[j]
                filescores[j] = temp
                temp1 = files[i]
                files[i] = files[j]
                files[j] = temp1

    rlist = []
    for i in range(n):
        rlist.append(files[i])


    return(rlist)    


def fliparray(array):
    narray = []
    for i in array:
        narray.append(0 - i)
    return(narray)    
def top_sentences(query, sentences, idfs, n):
    #returns the n most relavent sentences 

    sscores = []
    sqtd = []
    slist = list(sentences.keys())
    for i in slist:
        score = 0
        qterms = 0
        doneqterms = []
        for j in sentences[i]:
            if j in query:
                if not j in doneqterms and not j in QUESTION_WORDS:
                    qterms += 1
                    doneqterms.append(j)
                    score += idfs[j] 

                           

        qterms /= len(sentences[i])        
        sscores.append(score)
        sqtd.append(qterms)


    sscores = fliparray(sscores)    
    for i in range(len(sscores)):
        for j in range(len(sscores) - 1):
            if sscores[j] > sscores[j + 1]:
                temp = sscores[j + 1]
                sscores[j + 1] = sscores[j]
                sscores[j] = temp
                ntemp = slist[j]
                slist[j] = slist[j + 1]
                slist[j + 1] = ntemp
            elif sscores[j] == sscores[j + 1]:
                if sqtd[j] < sqtd[j + 1]:
                    temp = sscores[j]
                    sscores[j] = sscores[j + 1]
                    sscores[j + 1] = temp
                    ntemp = slist[j]
                    slist[j] = slist[j + 1]
                    slist[j + 1] = ntemp

    ranked = list(reversed(slist))
    rscores = list(reversed(sscores))
    outlist = []
    for i in range(n):
        outlist.append(ranked[i])
    return(outlist)                    
if __name__ == "__main__":
    main()
