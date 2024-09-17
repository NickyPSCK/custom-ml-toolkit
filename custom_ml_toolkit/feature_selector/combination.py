import itertools
from copy import deepcopy
from math import comb


class CombinatoricFeatureGenerator:
    def __init__(
        self,
        r_start: int,
        r_end: int,
        selected_cols: list,
        required_cols: list = None,
        budget: int = -1
    ):
        self._r_start = r_start
        self._r_end = r_end
        self._selected_cols = deepcopy(selected_cols)

        if required_cols is not None:
            self._required_cols = required_cols
        else:
            self._required_cols = list()

        self._budget = budget

        self._n = len(self._selected_cols)
        self._melted_required_cols = self.melt_list_of_list(required_cols)
        self._cal_range = self._create_cal_range()
        self._combinations = self._create_combinations()
        self._number_of_combinations = self._cal_number_of_combinations()
        self._remaining = self._number_of_combinations

        self._remaining_budget = budget

    @staticmethod
    def melt_list_of_list(list_of_list: list):
        result_list = list()
        for member in list_of_list:
            if isinstance(member, list):
                result_list += CombinatoricFeatureGenerator.melt_list_of_list(member)
            else:
                result_list.append(member)
        return result_list

    def _create_cal_range(self):
        if self._r_start <= self._r_end:
            cal_range = range(self._r_start, self._r_end + 1, 1)
        else:
            cal_range = range(self._r_start, self._r_end - 1, -1)
        return cal_range

    def _cal_number_of_combinations(self):
        no_of_cases = 0
        for r in self._cal_range:
            no_of_cases += comb(self._n, r)
        return no_of_cases

    @property
    def number_of_combinations(self):
        return self._number_of_combinations

    @property
    def budget(self):
        return self._budget

    @property
    def n(self):
        return self._n

    @property
    def remaining(self):
        return self._remaining

    @budget.setter
    def budget(self, budget: int):
        self._budget = budget
        self._remaining_budget = budget

    def info(self):
        print('---------------------------------------------------------------')
        print(f'n: {self.n}')
        for r in self._cal_range:
            print(f'r = {r}: {comb(self.n , r)} combinations')
        print(f'Remaining {self.remaining}/{self.number_of_combinations}')
        print('---------------------------------------------------------------')

    def _create_combinations(self):
        for r in self._cal_range:
            combinations = itertools.combinations(
                iterable=self._selected_cols,
                r=r
            )
            for combination in combinations:
                yield combination

    def _find_remove_members(self, combination):
        removed_members = list()
        for member in self._selected_cols:
            if member not in combination:
                removed_members.append(member)
        return removed_members

    def __next__(self):
        try:
            if self._budget > 0:
                if self._remaining_budget <= 0:
                    self._remaining_budget = self._budget
                    raise StopIteration()

            combination = next(self._combinations)
            removed_members = self._find_remove_members(combination)
            combination = self._melted_required_cols + list(combination)
            combination = self.melt_list_of_list(combination)
            combination = deepcopy(combination)

            self._remaining -= 1
            if self._budget > 0:
                self._remaining_budget -= 1

            return removed_members, combination

        except StopIteration:
            if self._remaining > 0:
                print(f'Run out of budget, but there is/are still {self._remaining} combination(s) left.')
            else:
                print('Run out of combination')
            raise StopIteration()

    def __iter__(self):
        return self

    def __len__(self):
        if self._budget > 0:
            return self._remaining_budget
        else:
            return self._remaining
