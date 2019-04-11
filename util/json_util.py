#!/usr/bin/env python
# -*- coding: utf-8 -*-

__AUTHOR__ = 'Roy'
__VERSION__ = '1.0.0'
__DATE_CREATED__ = '2015-9-23 08:00:01'
__DATE_MODIFIED__ = '2015-9-23 08:00:01'

from datetime import date, datetime
import json


def __default(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, date):
        return obj.strftime('%Y-%m-%d')
    else:
        raise TypeError('%r is not JSON serializable' % obj)


def json_dumps_zh(s, sort_keys=True):
    return json.dumps(s, ensure_ascii=False, sort_keys=sort_keys, default=__default)


def json_dumps(s, sort_keys=True):
    return json.dumps(s, separators=(',', ':'), sort_keys=sort_keys, default=__default)


def json_loads(s):
    return json.loads(s)
