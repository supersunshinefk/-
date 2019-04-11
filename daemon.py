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
# Author : Roy
# Description : 主程序守护者


#!/usr/bin/env python3
# daemon.py

import os
import sys
import time

import atexit
import signal

import config


class Daemon(object):
    # def __init__(self, project=None, program=None, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
    #     # 需要获取调试信息，改为stdin='/dev/stdin', stdout='/dev/stdout', stderr='/dev/stderr'，以root身份运行。
    #     if not project or not program:
    #         raise RuntimeError('project is null or program is null.')
    #
    #     pid_path = '{path}{project}/'.format(path=config.TEMP_PID_PATH, project=project)
    #     if not os.path.exists(pid_path):
    #         os.makedirs(pid_path)
    #
    #     pid_file = '{path}{program}.pid'.format(path=pid_path, program=program)
    #     stdout_file = '{path}{program}_stdout.log'.format(path=pid_path, program=program)
    #     stderr_file = '{path}{program}_stderr.log'.format(path=pid_path, program=program)
    #     self.pidfile = pid_file
    #     self.stdin = stdin
    #     self.stdout = stdout_file
    #     self.stderr = stderr_file

    def __init__(self, project=None, program=None):
        if not project or not program:
            raise RuntimeError('project is null or program is null.')

        pid_path = '{path}{project}/'.format(path=config.TEMP_PID_PATH, project=project)
        if not os.path.exists(pid_path):
            os.makedirs(pid_path)

        self.pid_file = '{path}{program}.pid'.format(path=pid_path, program=program)
        self.stdin = '/dev/null'
        self.stdout = '{path}{program}_stdout.log'.format(path=pid_path, program=program)
        self.stderr = '{path}{program}_stderr.log'.format(path=pid_path, program=program)

    def _daemonize_(self):
        if os.path.exists(self.pid_file):
            raise RuntimeError('{0} Already running'.format(self.pid_file))

        # First fork (detaches from parent)
        try:
            if os.fork() > 0:
                raise SystemExit(0)  # Parent exit
        except OSError as e:
            raise RuntimeError('fork #1 failed.', e)

        os.chdir('/')
        os.umask(0)
        os.setsid()
        # Second fork (relinquish session leadership)
        try:
            if os.fork() > 0:
                raise SystemExit(0)
        except OSError as e:
            raise RuntimeError('fork #2 failed.')

        # Flush I/O buffers
        sys.stdout.flush()
        sys.stderr.flush()

        # Replace file descriptors for stdin, stdout, and stderr
        with open(self.stdin, 'rb', 0) as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open(self.stdout, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
        with open(self.stderr, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stderr.fileno())

        # Write the PID file
        with open(self.pid_file, 'w') as f:
            print(os.getpid(), file=f)

        # Arrange to have the PID file removed on exit/signal
        atexit.register(lambda: os.remove(self.pid_file))

        # Signal handler for termination (required)
        def sigterm_handler(signo, frame):
            raise SystemExit(1)

        signal.signal(signal.SIGTERM, sigterm_handler)

    def del_pid(self):
        os.remove(self.pid_file)

    def start(self):
        try:
            self._daemonize_()
        except RuntimeError as e:
            print(e, file=sys.stderr)
            raise SystemExit(1)

        self.run()

    def stop(self):
        try:
            if os.path.exists(self.pid_file):
                with open(self.pid_file) as f:
                    os.kill(int(f.read()), signal.SIGTERM)
            else:
                print('Not running.', file=sys.stderr)
                raise SystemExit(1)
        except OSError as e:
            if 'No such process' in str(e) and os.path.exists(self.pid_file):
                os.remove(self.pid_file)

    def restart(self):
        self.stop()
        self.start()

    def run(self):
        pass
