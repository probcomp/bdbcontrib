# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import bayeslite
import os
import gzip
import json
import copy
from bayeslite.util import cursor_value
import re

_csv_fields = ["timestamp", "client_ip", "version", "id", "session_id", "type", "data", "start_time", "end_time", "error"]
keywords = ['select', 'simulate', 'infer', 'estimate', 'predict', 'given', 'predictive probability', 'probability of value', 'similarity', 'correlation', 'dependence probability', 'mutual information']

class _StoredSessionsEntryIterator:

    def __init__(self, sessions_dir, feature_extractors = {}):
        '''
            feeature_extractors is a dictionary of feature extractors for
            entries. the key will be the column name. the order of these
            columns in the resulting table is undefined
        '''
        
        feature_names = list(feature_extractors.keys())
        self.columns = _csv_fields + feature_names

        # iterate over the files in the directory
        files = []
        for (_, _, fnames) in os.walk(sessions_dir):
            files.extend(fnames)
            break
        self.csv_rows = []
        for file in files:
            print 'loading entries from %s ..' % (file,)
            if file.endswith('.json.gz'):
                with gzip.open(os.path.join(sessions_dir, file), 'rb') as f:
                    data = json.loads(f.read().decode("ascii"))
                    baserow = {}
                    baserow["timestamp"] = data["timestamp"]
                    baserow["client_ip"] = data["client_ip"]
                    baserow["version"] = data["values"]["version"]
                    fields = data["values"]["fields"]
                    entries = data["values"]["entries"]
                    for entry in entries:
                        row = copy.deepcopy(baserow)
                        for (field, value) in zip(fields, entry):
                            row[field] = value
                        # extract derived features
                        for (feat_name, feat) in feature_extractors.iteritems():
                            row[feat_name] = feat(row)
                        csv_str = ','.join([str(row[field]) for field in self.columns])
                        self.csv_rows.append(csv_str)
        self.csv_rows_idx = 0

    def __iter__(self):
        return self

    def next(self):
        if self.csv_rows_idx == 0:
            result = ','.join(self.columns)
        elif self.csv_rows_idx < len(self.csv_rows):
            result = self.csv_rows[self.csv_rows_idx]
        else:
            raise StopIteration()
        self.csv_rows_idx += 1
        return result

def primary_keyword(entry):
    return entry["data"].split(" ")[0]

def create_query_pattern_search(pattern):
    prog = re.compile(pattern, re.IGNORECASE)
    return lambda entry: prog.search(entry["data"]) is not None

def create_bdb_from_session_dumps(bdb_file, sessions_dir):

    if os.path.isfile(bdb_file):
        raise ValueError("The file %s exists, please use a different filename." % (bdb_file,))

    if not os.path.isdir(sessions_dir):
        raise ValueError("The provided path %s is not a directory." % (sessions_dir,))

    features = {}
    features["primary_keyword"] = primary_keyword

    # register feature extractors for each keyword
    for keyword in keywords:
        features[keyword] = create_query_pattern_search(keyword)
    entries_iter = _StoredSessionsEntryIterator(sessions_dir, features)

    bdb = bayeslite.bayesdb_open(bdb_file, builtin_metamodels=None)
    bayeslite.bayesdb_read_csv(bdb, 'entries', entries_iter, header=True, create=True, ifnotexists=True)
    num_entries = int(cursor_value(bdb.sql_execute("SELECT COUNT(*) FROM entries;")))
    print 'loaded %d entries' % (num_entries,)

    for row in list(bdb.sql_execute("SELECT primary_keyword,client_ip,dep_prob from entries;")):
        print row

# TODO: remove
create_bdb_from_session_dumps('sessions_test.bdb', 'dir')

