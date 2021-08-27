import unittest
import os
from gmbp_quant.static import get_project_root


class TestStatic(unittest.TestCase):
    def test_project_root(self):
        project_root = get_project_root()
        folders = {folder for folder in os.listdir(project_root)
                   if os.path.isdir(os.path.join(project_root, folder)) and '.' not in folder}
        self.assertEqual(folders, {'gmbp_quant', 'test', 'notebook', 'runtime_env'})
    #
#


if __name__ == '__main__':
    unittest.main()
#
