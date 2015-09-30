
import bayeslite
import bdbcontrib
from bdbcontrib import sessions

# TODO MAKE INTO AN ACTUAL TEST

# Load the satellites snapshot into a local instance of bayeslite

bdb = bayeslite.bayesdb_open(pathname='satellites.bdb')

# Define a utility procedure to conveniently query satellites_bdb
def q(query_string):
    res = bdb.execute(query_string)
    return bdbcontrib.cursor_to_df(res)

#q("SELECT * FROM satellites WHERE Name LIKE 'International Space Station%'").transpose()


#print q("select * from bayesdb_session;")
#print 'session 7'
#print q("select * from bayesdb_session_entries where session_id=7;")
#print 'session 8'
#print q("select * from bayesdb_session_entries where session_id=8;")
#print 'session 9'
#print q("select * from bayesdb_session_entries where session_id=9;")

#print 'current session: ', bql_utils.current_session_id(satellites_bdb)
print q('select * from bayesdb_session_entries where completed=0 order by time')

#q('drop table unlikely_periods')

#q('''CREATE TEMP TABLE unlikely_periods AS ESTIMATE name, class_of_orbit, period_minutes,
#PREDICTIVE PROBABILITY OF period_minutes AS "Relative Probability of Period"
#FROM satellites_cc;
#''')

print 'list sessions'
print sessions.list_sessions(bdb)

print 'current session id'
print sessions.current_session_id(bdb)

print 'dump session as json 1'
print sessions.dump_session_as_json(bdb, 1)

print 'dump current session as json'
print sessions.dump_current_session_as_json(bdb)

print 'dump session 2 as json'
print sessions.dump_session_as_json(bdb, 1)

print 'delete all entries'
bdb.sql_execute('DELETE FROM bayesdb_session_entries;')

print 'select all entries'
print q('select * from bayesdb_session_entries;')

#print 'clear all session data'
print sessions.clear_all_sessions(bdb)

print 'list all sessions'
print bdbcontrib.cursor_to_df(sessions.list_sessions(bdb))

print 'list all entries'
print bdbcontrib.cursor_to_df(bdb.sql_execute('select id,session_id from bayesdb_session_entries;'))

print 'stop saving session entries'
sessions.stop_saving_session_entries(bdb)

print 'doing some stuff'
q('select * from bayesdb_session_entries where completed=0 order by time')
q('select * from bayesdb_session_entries where completed=0 order by time')
q('select * from bayesdb_session_entries where completed=0 order by time')

print 'list all entries (should be no change)'
print bdbcontrib.cursor_to_df(bdb.sql_execute('select id,session_id from bayesdb_session_entries;'))

print 'start saving session entries'
sessions.start_saving_session_entries(bdb)

print 'doing some stuff'
q('select * from bayesdb_session_entries where completed=0 order by time')
q('select * from bayesdb_session_entries where completed=0 order by time')
q('select * from bayesdb_session_entries where completed=0 order by time')


print 'list all entries (shoud be more stuff)'
print bdbcontrib.cursor_to_df(bdb.sql_execute('select id,session_id from bayesdb_session_entries;'))

