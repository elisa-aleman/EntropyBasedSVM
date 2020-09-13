#-*- coding: utf-8 -*-

import os
import sys
import codecs
import csv
import numpy
import pandas
from io import StringIO

######################################
###### Special printing methods ######
######################################

def up():
    '''
    Go up a line in the terminal to print over something
    '''
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()

def down(): 
    '''
    Go down a line in the terminal (like print(""))
    '''
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()

def print_STD_log(strlog, log_file): 
    '''
    Print string to terminal and to file at the same time
    '''
    with codecs.open(log_file, 'a', 'utf-8') as logf:
        print(strlog)
        strlog+= "\n"
        logf.write(strlog)

def print_log(strlog, log_file):
    '''
    Print string to a file
    '''
    with codecs.open(log_file, 'a', 'utf-8') as logf:
        strlog+= "\n"
        logf.write(strlog)

def print_list_to_log(strlist, log_file):
    '''
    Print list of strings to a file
    '''
    with codecs.open(log_file, 'a', 'utf-8') as logf:
        for strlog in strlist:
            strlog+= "\n"
            logf.write(strlog)

def print_list_to_STD_log(strlist, log_file):
    '''
    Print list of strings to the terminal and to a file at the same time
    '''
    with codecs.open(log_file, 'a', 'utf-8') as logf:
        for strlog in strlist:
            print(strlog)
            strlog+= "\n"
            logf.write(strlog)
        
def write_tuple_list_to_csv(ins_table, filepath):
    '''
    Write tuple list to CSV file
    '''
    with open(filepath,'w') as out:
        csv_out=csv.writer(out, delimiter=',', quotechar='"', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        csv_out.writerows(ins_table)

def append_tuple_to_csv(ins_tuple, filepath):
    '''
    Append tuple to CSV file
    '''
    with open(filepath,'a+') as out:
        csv_out=csv.writer(out, delimiter=',', quotechar='"', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        csv_out.writerow(ins_tuple)
        
################################################
###### Data reading / organizing methods #######
################################################

def tail(filepath, decode_utf8=True, with_head=True, pandas_read=True):
    '''
    Returns the last line in a file
    '''
    with open(filepath, "rb") as f:
        first = f.readline()      # Read the first line.
        f.seek(-2, 2)             # Jump to the second last byte.
        while f.read(1) != b"\n": # Until EOL is found...
            try:
                f.seek(-2, 1)     # ...jump back the read byte plus one more.
            except IOError:
                f.seek(-1, 1)
                if f.tell() == 0:
                    break
        last = f.readline()       # Read last line.
    if decode_utf8:
        if with_head:
            first = first.decode('utf-8')
        last = last.decode('utf-8')
    if with_head:
        last = first+last
    if pandas_read:
        last = [tuple(row) for row in csv.reader(StringIO(last), delimiter=',', quotechar='"', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)]
        if with_head:
            last = pandas.DataFrame([last[1]], columns=last[0])
        else:
            last = pandas.DataFrame(last)
    return last


def head(filepath, decode_utf8=True):
    '''
    Returns the first line in a file
    '''
    with open(filepath, "rb") as f:
        first = f.readline()
    if decode_utf8:
        first = first.decode('utf-8')
    return first

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def line_count_rawgen(filepath, last_index_csv=True):
    '''
    Fast way to count lines in a file
    https://stackoverflow.com/a/27518377
    '''
    f = open(filepath, 'rb')
    f_gen = _make_gen(f.raw.read)
    if last_index_csv:
        return sum( buf.count(b'\n') for buf in f_gen ) - 2
    else:
        return sum( buf.count(b'\n') for buf in f_gen )

def read_dict(filename):
    '''
    Get a word list text file (a word per line) into a list
    '''
    dictionary = []
    with open(filename, 'r') as thefile:
        for line in thefile:
            if (line!=""):
                nline = line.replace('\n','')
                dictionary.append(nline)
    return dictionary

def flatten(container):
    '''
    Make a list or tuple like [1,[2,3],[4,[5]]] into [1,2,3,4,5]
    '''
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

if __name__ == '__main__':
    pass