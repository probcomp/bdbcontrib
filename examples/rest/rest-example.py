#!/usr/bin/env python

import argparse
import flask
from flask import request
import json
import jsonschema
import logging
import os

import bayeslite
import bdbcontrib as bdb

# TODO: How to pass instance path as an arg?
app = flask.Flask(__name__, instance_path=os.getcwd())
args = None

# TODO: This is in a function because flask was opening it in another
# thread. Maybe do some kind of thread local cache or otherwise figure
# out how to not leak these.
def getbdb():
    app.logger.info("Opening database %s" % (args.bdbpath,))
    return bayeslite.bayesdb_open(args.bdbpath)

@app.route('/')
def root():
    try:
        with open('static/index.html') as f:
            return f.read()
    except:
        return flask.make_response("Run in a directory with a static/index.html and other static resources in static/.", 500)

@app.route('/favicon.ico')
def favicon():
    try:
        with open('static/favicon.ico') as f:
            r = flask.make_response(f.read, 200)
            r.mimetype = 'image/vnd.microsoft.icon'
            return r
    except:
        return flask.make_response("No favicon", 404)

def cursor_to_response(cursor):
    j = bdb.cursor_to_df(cursor).to_json(orient="records")
    r = flask.make_response(j, 200)
    r.mimetype = 'text/json'
    return r

@app.route('/api/v1/bql')
def bql():
    with getbdb() as bdb:
        return cursor_to_response(bdb.execute(request.values['q'].encode('ascii', 'ignore')))

@app.route('/api/v1/sql')
def sql():
    with getbdb() as bdb:
        return cursor_to_response(bdb.sql_execute(request.values['q'].encode('ascii', 'ignore')))

@app.route('/api/v1/table', defaults={'tablename': None})
@app.route('/api/v1/table/<tablename>')
def table(tablename):
    with getbdb() as bdb:
        if tablename is None:
            return cursor_to_response(bdb.sql_execute("""
                  select * from sqlite_master
                  where type = 'table'
                  and tbl_name not like 'bayesdb_%'
                  and tbl_name not like 'sqlite_%'"""))
        else:
            return cursor_to_response(bdb.describe_table(bdb, request.values['table']))

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('bdbpath', type=str,
                        help="bayesdb database file")
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help="Port to listen on")
    parser.add_argument('--host', default="127.0.0.1",
                        help="Address to bind.")
    parser.add_argument('-l', '--loglevel', type=int, default=logging.INFO,
                        help="Set the log level.")
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    app.logger.addHandler(logging.StreamHandler())
    app.logger.setLevel(args.loglevel)
    app.run(host=args.host, port=args.port, debug=True)
