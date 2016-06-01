import json
import jsonschema
import pkgutil

def test_example_schema():
    schema_json = pkgutil.get_data('bdbcontrib', 'mml.schema.json')
    schema = json.loads(schema_json)
    example = json.load(open('tests/example_mml_schema.json'))
    jsonschema.validate(example, schema)
