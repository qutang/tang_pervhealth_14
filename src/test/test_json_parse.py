
import json
import numpy

s = '{"DW":{"mean":"test"}}'
d = json.loads(s)

json_str = open('test.json').read()
print json_str
parsed_dict = json.loads(json_str)
print parsed_dict['DW']['mean']['apply'][0]

func = getattr(numpy, "mean")
print func(range(1,10))