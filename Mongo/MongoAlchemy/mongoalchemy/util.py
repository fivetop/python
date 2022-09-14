# The MIT License
#
# Copyright (c) 2010 Jeffrey Jenkins
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function
from mongoalchemy.py3compat import *

def classproperty(fun):
    class Descriptor(property):
        def __get__(self, instance, owner):
            return fun(owner)
    return Descriptor()

class UNSET(object):
    def __repr__(self):
        return 'UNSET'
    def __eq__(self, other):
        return other.__class__ == self.__class__
UNSET = UNSET()

class FieldNotFoundException(Exception):
    pass

def resolve_name(type, name):
    if not isinstance(name, basestring) or name[0] == '$':
        return name
    ret = type
    for part in name.split('.'):
        try:
            ret = getattr(ret, part)
        except AttributeError:
            raise FieldNotFoundException("Field not found %s (in %s)" %
                                         (part, name))

    return ret

