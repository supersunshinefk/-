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
# Author :XinJinquan
# Description : filter过滤器脚本

### 以下 a是原始来的数据， b是脚本中的值

def greater_than(a, b):

    if a > b:
        return True


def not_equal(a, b):
    if a != b:
        return True


def greater_or_equal(a, b):
    if a >= b:
        return True


def less_than(a, b):
    if a <= b:
        return True


def all(a, b):
    return False


def less_or_equal(a, b):
    if a <= b:
        return True


def equal(a, b):
    if a == b:
        return True

# ["L1_3_4", "@equal", 0],
# 			["L1_3_15", "@not_equal", 0],
# 			["L1_3_16", "@greater_than", 0],
# 			["L1_3_38", "@less_than", 0],
# 			["L1_3_39", "@greater_or_equal", 0],
# 			["L1_3_41", "@less_or_equal", 0]

