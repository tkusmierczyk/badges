#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import logging


def run(method, *args, **kwargs):
    label = kwargs.pop("method_label", str(method))
    logging.info("[timer] calling: %s(%s, %s)" % 
          (label, str(args)[:100].replace("\n", "    "), str(kwargs)[:100].replace("\n", "    ")))
    start = time.time()
    result = method(*args, **kwargs)
    end = time.time()
    logging.info("[timer] %s used: %.4fs" % (label, end-start))
    return result

