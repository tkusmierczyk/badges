#!/usr/bin/python3
# -*- coding: utf-8 -*-



def str2value(s):
    original = s
    s = s.lower()
    if s=="t" or s=="true": return True
    if s=="f" or s=="false": return False    
    try: 
        if int(s)==float(s): return int(s)
    except: pass    
    try: return float(s)
    except: pass
    return original


def parse_dictionary(options_str):
    options_str = options_str.replace(";", ",").replace(":", "=")
    options = [o.strip() for o in options_str.split(",") if len(o.strip())>0]
    options_dict = {}
    for option in options:
        if "=" not in option: 
            raise ValueError("options must be given as option=value") 
        parts = option.split("=")
        option, val = parts[0], parts[1]
        options_dict[option] = str2value(val)
    return options_dict


def format_dict(dct):
    return str(dct).strip("{").strip("}").replace("\"", "").replace("'", "")


class objectview(object):
    
    def __init__(self, d):
        self.d = d.copy()
        self.__dict__ = self.d
                
    def __str__(self):
        return(str(self.d))