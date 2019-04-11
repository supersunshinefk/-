#!/usr/bin/env python
# -*- coding: utf-8 -*-

__AUTHOR__ = 'Roy'
__VERSION__ = '1.0.0'
__DATE_CREATED__ = '2015-10-24 08:00:01'
__DATE_MODIFIED__ = '2015-10-24 08:00:01'

from urllib import parse
import httplib2
import base64

from util.json_util import json_loads

lam_string_idx = (lambda content, s: content.find(s) if content.find(s) >= 0 else 0)
lam_string_ridx = (lambda content, s: content.rfind(s) + len(s) if content.rfind(s) > 0 else len(content))


class HttpHelper():

    def __init__(self):
        self.REQ_TIMEOUT = 15
        pass

    def make_basic_auth(self, username, password):
        temp = username + ':' + password
        _hash = base64.b64encode(temp.encode('utf-8'))
        return "Basic " + str(_hash, 'utf-8')

    def postdata(self, req_url, post_params, auth_username=None, auth_password=None):
        """
            req_url url
            data    {'param1':"value1",'param2':"value2"}
        """
        resp, content = False, None
        try:
            h = httplib2.Http(timeout=self.REQ_TIMEOUT)
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            if auth_username and auth_password:
                headers["Authorization"] = self.make_basic_auth(auth_username, auth_password)
            # print 'request url:%s,data:%s' % (req_url, post_params)
            if post_params is None or len(post_params) == 0:
                resp, content = h.request(req_url, "POST", None, headers=headers)
            else:
                resp, content = h.request(req_url, "POST", parse.urlencode(post_params), headers=headers)
            if resp and resp.status == 200 and content:
                content = content.decode()
                # print 'response content:%s' % (content)
                idx = content.find("{")
                if idx >= 0:
                    r_idx = lam_string_ridx(content, "}")
                    content = content[idx:r_idx]
                    result = json_loads(content)
                    if result and len(result) > 0:
                        return True, content
            else:
                # print('response resp:%s,content:%s' % (resp, content))
                pass
        except Exception as e:
            print('request error:%s' % e)

        return False, content

    def getdata(self, req_url, get_params, auth_username=None, auth_password=None):
        """
            req_url url
            data    {'param1':"value1",'param2':"value2"}
        """
        resp, content = False, None
        try:
            h = httplib2.Http(timeout=self.REQ_TIMEOUT)
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            if auth_username and auth_password:
                headers["Authorization"] = self.make_basic_auth(auth_username, auth_password)

            if get_params and len(get_params) > 0:
                url_params = parse.urlencode(get_params)
                idx = req_url.find("?")
                if req_url.find("?") < 0:
                    req_url = req_url + "?" + url_params
                else:
                    if (idx + 1) == len(req_url):
                        req_url = req_url + url_params
                    else:
                        req_url = req_url + "&" + url_params
            # print 'request url:%s,url_params:%s' % (req_url, url_params)
            resp, content = h.request(req_url, "GET", headers=headers)
            if resp and resp.status == 200 and content:
                content = content.decode()
                # print 'response content:%s' % (content)
                idx = content.find("{")
                if idx >= 0:
                    r_idx = lam_string_ridx(content, "}")
                    content = content[idx:r_idx]
                    result = json_loads(content)
                    if result and len(result) > 0:
                        return True, content
            else:
                # print('response resp:%s,content:%s' % (resp, content))
                pass
        except Exception as e:
            print('request error:%s' % e)

        return False, content
