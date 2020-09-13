#-*- coding: utf-8 -*-

####################################
########## Training Data  ##########
####################################

def PosiNegaSentences(comment):
    '''
    Input is "word word <positive> word word word </positive> <negative> word word word </negative> word ..."
    
    Returns a list of sentences in the format ['word word word',1] if positive or ['word word word', 0] if negative
    '''
    comment = comment.split()
    # The comment has been reduced to a list of words including the tags
    sentences = []
    sentence = []
    att = 0
    positive = False
    negative = False
    for current_word in comment:
        skip = False
        if (current_word == '<positive>'):
            positive = True
            negative = False
            skip = True
        if (current_word == '</positive>'):
            positive = True
            skip = True
        if (current_word == '<negative>'):
            positive = False
            negative = True
            skip = True
        if (current_word == '</negative>'):
            negative = True
            skip = True
        if not skip:
            if positive or negative:
                sentence.append(current_word)
        else:
            if len(sentence)>0:
                if positive and not negative:
                    att = 1
                    sentence = " ".join(sentence).strip()
                    sentences.append([sentence,att])
                    positive = False
                elif negative and not positive:
                    att = 0
                    sentence = " ".join(sentence).strip()
                    sentences.append([sentence,att])
                    negative = False
                sentence = []
    return sentences

def PosiNegaNeutraSentences(comment):
    '''
    Input is "<neutral> word word </neutral> <positive> word word word </positive> <negative> word word word </negative> word ..."
    
    Returns a list of sentences in the format 
       ['word word word',  1] if positive
    or ['word word word',  0] if neutral
    or ['word word word', -1] if negative
    '''
    comment = comment.split()
    # The comment has been reduced to a list of words including the tags
    sentences = []
    sentence = []
    att = 0
    positive = False
    negative = False
    neutral = False
    for current_word in comment:
        skip = False
        if (current_word == '<positive>'):
            positive = True
            negative = False
            neutral = False
            skip = True
        if (current_word == '</positive>'):
            positive = True
            skip = True
        if (current_word == '<negative>'):
            positive = False
            neutral = False
            negative = True
            skip = True
        if (current_word == '</negative>'):
            negative = True
            skip = True
        if (current_word == '<neutral>'):
            positive = False
            neutral = True
            negative = False
            skip = True
        if (current_word == '</neutral>'):
            neutral = True
            skip = True
        if not skip:
            if positive or negative or neutral:
                sentence.append(current_word)
        else:
            if len(sentence)>0:
                if positive and not negative and not neutral:
                    att = 1
                    sentence = " ".join(sentence).strip()
                    sentences.append([sentence,att])
                    positive = False
                elif negative and not positive and not neutral:
                    att = -1
                    sentence = " ".join(sentence).strip()
                    sentences.append([sentence,att])
                    negative = False
                elif neutral and not positive and not negative:
                    att = 0
                    sentence = " ".join(sentence).strip()
                    sentences.append([sentence,att])
                    neutral = False
                sentence = []
    return sentences

if __name__ == '__main__':
    pass