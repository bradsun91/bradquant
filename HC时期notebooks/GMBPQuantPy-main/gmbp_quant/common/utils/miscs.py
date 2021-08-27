import numbers
import enum
from collections import OrderedDict
from typing import Iterable


def iterable_to_tuple(items, raw_type, sep=',', remove_duplicates=True):
    if isinstance(items, str):
        items = (item.strip() for item in items.split(sep))
        if remove_duplicates:
            items = OrderedDict.fromkeys(items).keys()
        #
    elif isinstance(items, numbers.Integral):
        items = (items, )
    elif not isinstance(items, Iterable):
        raise Exception(f'type(items)={type(items)} is not supported [str|numbers.Integral|typing.Iterable]')
    #

    raw_type = raw_type.lower()
    if raw_type == 'int':
        items = tuple(int(item) for item in items)
    elif raw_type == 'str':
        items = tuple(str(item) for item in items)
    else:
        raise Exception(f'raw_type={raw_type} is not supported [str|int]')
    #

    return items
#


def iterable_to_db_str(items, raw_type, sep=',', remove_duplicates=True):
    items = iterable_to_tuple(items=items, raw_type=raw_type, sep=sep, remove_duplicates=remove_duplicates)

    if len(items)==0:
        raise Exception('Param items should contain at least one element ！')
    #

    if raw_type == 'int':
        items = f"({','.join([str(item) for item in items])})"
    else:
        items = "','".join(items)
        items = f"('{items}')"
    #

    return items
#


class Enum(enum.Enum):
    @classmethod
    def from_name(cls, val):
        if val is None:
            return None
        #

        for name, item in cls.__members__.items():
            if name.upper() == val.upper():
                return item
            #
        #

        raise Exception(f'Failed to find Enum with value={val} !')
    #
#
