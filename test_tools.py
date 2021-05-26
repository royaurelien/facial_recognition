import os
import json
import time
import unittest

# from .. import tools
import tools

class TestToolsGroupBy(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

        path = 'test_tools_dataset.json'

        with open(path) as f:
            self.dataset = json.load(f).get('dataset')

        for key in self.dataset.keys():
            self.dataset[key]['result'] = self.sort_result(self.dataset[key]['result'])
            self.dataset[key]['len'] = len(self.sort_result(self.dataset[key]['data']))
            # name = "test_{}".format(key)
            # def sample_method(self):
            #     res = tools.group_by_set(self.dataset[key]['data'])
            #     self.assertEqual(res, self.dataset[key]['result'])
            # self.__dict__[name] = sample_method.__get__(self)

        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))

    def sort_result(self, data):
        return sorted([sorted(items) for items in data])

    def get_result_of(self, serie, func, *args, **kwargs):
        print(">> Call {}() with {} ({} elements)".format(func.__name__, serie, self.dataset[serie]['len']))
        res = func(self.dataset[serie]['data'], *args, **kwargs)
        self.assertEqual(res, self.dataset[serie]['result'])

    def test_count_group_by_set(self):
        res = tools.group_by_set(self.dataset['serie_1']['data'])
        self.assertCountEqual(res, self.dataset['serie_1']['result'])

    def test_dataset_1_group_by_set(self):
        self.get_result_of('serie_1', tools.group_by_set)

    def test_dataset_2_group_by_set(self):
        self.get_result_of('serie_2', tools.group_by_set)

    def test_dataset_3_group_by_set(self):
        self.get_result_of('serie_3', tools.group_by_set)

    def test_dataset_4_group_by_set(self):
        self.get_result_of('serie_4', tools.group_by_set)

    def test_count_group_by_list(self):
        res = tools.group_by_list(self.dataset['serie_1']['data'])
        self.assertCountEqual(res, self.dataset['serie_1']['result'])

    def test_dataset_1_group_by_list(self):
        self.get_result_of('serie_1', tools.group_by_list)

    def test_dataset_2_group_by_list(self):
        self.get_result_of('serie_2', tools.group_by_list)

    def test_dataset_3_group_by_list(self):
        self.get_result_of('serie_3', tools.group_by_list)

    # def test_group_by_list_dataset_4(self):
    #     self.get_result_of('serie_4', tools.group_by_list)

if __name__ == '__main__':
    unittest.main()