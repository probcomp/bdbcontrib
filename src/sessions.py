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

csv_fields = ["timestamp", "client_ip", "version", "id", "session_id", "type", "data", "start_time", "end_time", "error"]

class Entries:

    def __init__(self, sessions_dir):
        # iterate over the files in the directory
        files = []
        for (_, _, fnames) in os.walk(sessions_dir):
            files.extend(fnames)
            break
        csv_rows = []
        for file in files:
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
                            row[field] = str(value)
                        csv_str = ','.join([row[field] for field in csv_fields])
                        csv_rows.append(csv_str)
        csv_rows_idx = 0

    def next():
        if csv_rows_idx == 0:
            result = ','.join(csv_fields)
        elif csv_rows_idx < len(csv_rows):
            result = csv_rows[csv_rows_idx]
        else:
            return None
        csv_rows_idx += 1



def create_bdb_from_session_dumps(bdb_file, sessions_dir):

    entries_iter = Entries(sessions_dir)
    bdb = bayeslite.bayesdb_open(bdb_file, builtin_metamodels=None)
    bayeslite.bayesdb_read_csv(bdb, 'entries', entries_iter, header=True, create=True, ifnotexists=False)

create_bdb_from_session_dumps('sessions_test.bdb', '/afs/csail/proj/probcomp/bayeslite_saved_sessions')
