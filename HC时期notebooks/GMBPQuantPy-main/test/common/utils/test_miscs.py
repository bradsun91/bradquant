import pandas as pd
import unittest
import gmbp_quant.common.utils.miscs as mu


class TestMiscs(unittest.TestCase):
    def test_iterable_to_tuple(self):
        input = 'Hello'
        output = mu.iterable_to_tuple(input, raw_type='str')
        self.assertEqual(output, ('Hello',))

        input = '5'
        output = mu.iterable_to_tuple(input, raw_type='int')
        self.assertEqual(output, (5,))

        input = 'Hello, World'
        output = mu.iterable_to_tuple(input, raw_type='str')
        self.assertEqual(output, ('Hello', 'World'))

        input = '5,3,-1,3,5,9'
        output = mu.iterable_to_tuple(input, raw_type='int')
        self.assertEqual(output, (5, 3, -1, 9))

        input = '5,3,-1,3,5,9'
        output = mu.iterable_to_tuple(input, raw_type='int', remove_duplicates=False)
        self.assertEqual(output, (5, 3, -1, 3, 5, 9))

        input = '5, -1,    3,9,9'
        output = mu.iterable_to_tuple(input, raw_type='int')
        self.assertEqual(output, (5, -1, 3, 9))

        input = pd.Series([4, -2, 9, 'test'])
        output = mu.iterable_to_tuple(input, raw_type='str')
        self.assertEqual(output, ('4', '-2', '9', 'test'))

        input = -4
        output = mu.iterable_to_tuple(input, raw_type='int')
        self.assertEqual(output, (-4,))
    #

    def test_iterable_to_db_str(self):
        input = '5,3,-1,3,5,9'
        output = mu.iterable_to_db_str(input, raw_type='int')
        self.assertEqual(output, '(5,3,-1,9)')

        input = 'Hello, World'
        output = mu.iterable_to_db_str(input, raw_type='str')
        self.assertEqual(output, "('Hello','World')")
    #

    def test_Enum(self):
        class TestRegionEnum(mu.Enum):
            AP = 1
            EU = 2
            US = 3
            WW = 4
        #

        self.assertEqual(TestRegionEnum.from_name('AP'), TestRegionEnum.AP)
        self.assertEqual(TestRegionEnum.from_name('EU'), TestRegionEnum.EU)
        self.assertEqual(TestRegionEnum.from_name('US'), TestRegionEnum.US)
        self.assertEqual(TestRegionEnum.from_name('WW'), TestRegionEnum.WW)
    #
#


if __name__ == '__main__':
    unittest.main()
#
