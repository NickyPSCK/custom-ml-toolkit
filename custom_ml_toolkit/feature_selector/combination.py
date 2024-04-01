import itertools
from copy import copy, deepcopy
from math import comb

class ConbinatoricFeatureGenerator:
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

        self._melted_required_cols = self.melt_list_of_list(required_cols)
        self._cal_range = self._create_cal_range()
        self._combinations = self._create_combinations()
        self._remaining = self.number_of_cases

        self._remaining_budget = budget

    @staticmethod
    def melt_list_of_list(list_of_list: list):
        result_list = list()
        for member in list_of_list:
            if isinstance(member, list):
                result_list += ConbinatoricFeatureGenerator.melt_list_of_list(member)
            else:
                result_list.append(member)
        return result_list
    
    def _create_cal_range(self):
        if self._r_start <= self._r_end:
            cal_range = range(self._r_start, self._r_end + 1, 1)
        else:
            cal_range = range(self._r_start, self._r_end - 1, -1)
        return cal_range

    @property
    def number_of_cases(self):
        no_of_cases = 0
        for r in self._cal_range:
            no_of_cases += comb(len(self._selected_cols), r)
        return no_of_cases
    
    @property
    def budget(self):
        return self._budget

    @property
    def remaining(self):
        return self._remaining
    
    @budget.setter
    def budget(self, budget):
        self._budget = budget
        self._remaining_budget = budget


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
                # print(member, combination)
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
                print(f'Run out of combination')
            raise StopIteration()

    def __iter__(self):
        return self

if __name__ == '__main__':
    r_start = 1
    r_end = 3
    required_cols = ['r_1', 'r_2', ['r_3', 'r_4']]
    selected_cols = ['e_1', 'e_2', ['e_3_1', 'e_3_2']]
    budget = 1
    
    cfg = ConbinatoricFeatureGenerator(
        r_start=r_start,
        r_end=r_end,
        selected_cols=selected_cols,
        required_cols=required_cols,
        budget=budget
    )
    
    print(cfg.number_of_cases)
    
    for removed_members, combination in cfg:
        print(cfg.remaining, removed_members, combination)
    
    cfg.budget = 2
    
    for removed_members, combination in cfg:
        print(cfg.remaining, removed_members, combination)
    
    cfg.budget = -1
    
    for removed_members, combination in cfg:
        print(cfg.remaining, removed_members, combination)
