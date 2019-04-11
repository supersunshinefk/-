#!/usr/bin/local/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright (c) 2014-2018 Mixlinker Networks Inc. <mixiot@mixlinker.com>
# All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Application License of Mixlinker Networks License and Mixlinker
# Distribution License which accompany this distribution.
#
# The Mixlinker License is available at
#    http://www.mixlinker.com/legal/license.html
# and the Mixlinker Distribution License is available at
#    http://www.mixlinker.com/legal/distribution.html
#
# Contributors:
#    Mixlinker Technical Team
###############################################################################
# Date : 2018-5-29
# Author :Roy
# Description : indass管理程序

__AUTHOR__ = 'Roy'
__VERSION__ = '1.0.0'
__DATE_CREATED__ = '2018-4-8 19:55:25'
__DATE_MODIFIED__ = '2018-4-8 19:55:25'

import os, sys
import subprocess
import json
import signal

import config


def start_project(project_queue_id):
    """
    :param project_queue_id: redis中保存的project队列id
    :return:
    """
    for program in config.PROGRAM:
        p_args = [config.INTERPRETER, program, project_queue_id]
        subprocess.call(p_args)


def stop_project(project_id):
    """
    :param project_id: redis中保存的project id
    :return:
    """
    try:
        for program in config.ProgramEnum:
            pid_file = '{path}{project}/{program}.pid'.format(path=config.TEMP_PID_PATH, project=project_id,
                                                              program=program.name)
            if os.path.exists(pid_file):
                with open(pid_file) as f:
                    os.kill(int(f.read()), signal.SIGTERM)
                    print('kill {pid_file}.'.format(pid_file=pid_file))
            else:
                print('{pid_file} Not running.'.format(pid_file=pid_file), file=sys.stderr)
                # raise SystemExit(1)
    except OSError as e:
        print('{pid_file} Error,{err}.'.format(pid_file=pid_file, err=e), file=sys.stderr)


def main():
    while True:
        try:
            res = config.redis_conn.blpop(config.LIST_BOSS_PROJECT)
            project_queue_id = res[1]
            project_queue_id = project_queue_id.decode("utf-8")
            print("\n")
            print("project_queue_id:", project_queue_id)
            res = config.redis_conn.hget(config.HSET_BOSS_PROJECT, project_queue_id)
            if res:
                # str(d, encoding='utf-8')
                res = res.decode("utf-8")
                project_info = json.loads(res)
                if project_info:
                    project_id = project_info["project_id"]
                    project_status = project_info["server"]
                    print(">>>listen,{project_id}={status}".format(project_id=project_id, status=project_status))
                    if "on" == project_status:
                        start_project(project_queue_id)
                        pass
                    else:
                        stop_project(project_id)
                        pass
        except Exception as ex:
            print('py_manager ERROR = %s' % ex)
            continue


if __name__ == "__main__":
    main()
