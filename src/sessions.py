import json

def list_sessions(bdb):
    """Lists all saved sessions with the number of entries in each, and whether
    they were sent or not."""
    return bdb.execute('select * from bayesdb_session;')


def current_session_id(bdb):
    """Returns the current integer session id."""
    return bdb.session_id


def dump_session_as_json(bdb, session_id):
    """Returns a JSON string representing the list of SQL or BQL entries (e.g.
    queries) executed within session `session_id`."""
    current_session = current_session_id(bdb)
    if session_id > current_session or session_id < 1:
        raise ValueError('No such session (%d)' % session_id)
    bql = """SELECT * FROM bayesdb_session_entries
    WHERE session_id == ?
    ORDER BY time DESC"""
    entries = bdb.sql_execute(bql, (session_id,))
    return json.dumps([entry for entry in entries])


def dump_current_session_as_json(bdb):
    """Returns a JSON string representing the current sesion (see
    `dump_session_as_json`)"""
    return dump_session_as_json(bdb, current_session_id(bdb))


def clear_all_sessions(bdb):
    bdb.clear_all_sessions()


def send_session_data(bdb):
    """Send all saved session history."""
    session_id = current_session_id(bdb)
    for id in range(1, session_id+1):
        json_string = dump_session_as_json(bdb, id)
        # TODO send it


def stop_saving_session_entries(bdb):
    """Turn off saving of session entries"""
    bdb.unset_save_sessions()
    

def start_saving_session_entries(bdb):
    """Turn on saving of session entries"""
    bdb.set_save_sessions()


