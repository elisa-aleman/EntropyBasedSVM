#-*- coding: utf-8 -*-

import sqlite3

##########################################
############## CSV and SQL ###############
##########################################

def Connect(dbname):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    return conn,c

def CSVcreateSQL(titles, dbname, tablename):
    conn, c = Connect(dbname)
    size = len(titles)
    # titles_quotes = ["'{}'".format(i) for i in titles]
    # titles_str = ", ".join(titles_quotes)
    c.execute("CREATE TABLE {tn} ('{nf}' INTEGER PRIMARY KEY)".format(tn=tablename, nf=titles[0]))
    for title in titles[1:]:
        c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' INTEGER".format(tn=tablename, cn=title))
        conn.commit()
    conn.commit()

def CSVtoSQL(titles, table, dbname, tablename='users'):
    times = len(titles)/100
    conn, c = Connect(dbname)
    columns = titles
    columns_quote = ["'{}'".format(i) for i in columns]
    columns_str = ", ".join(columns_quote)
    sentence = "?{}".format(",?"*(len(columns)-1))
    sql = "INSERT INTO {} ({}) VALUES({})".format(tablename, columns_str, sentence)
    ins = [tuple(row) for row in table]
    c.executemany(sql,ins)
    conn.commit()
    conn.close()

def SQLtoCSV(dbname):
    # dbname = 'ctrip_db2.sqlite'
    conn, c = Connect(dbname)
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    tables = [table[0] for table in tables]
    for table in tables:
        sql = u"SELECT * FROM {}".format(table)
        c.execute(sql)
        raw = c.fetchall()
        columns = [des[0] for des in c.description]
        log_file = MakeSQLpath(os.path.join('CSV',"{}.csv".format(table)))
        strlog = u",".join(columns)
        printLog(strlog, log_file)
        for row in raw:
            strlog = u",".join([u'"{}"'.format(item) if type(item) == type(u'') else u'{}'.format(item) for item in row])
            printLog(strlog, log_file)