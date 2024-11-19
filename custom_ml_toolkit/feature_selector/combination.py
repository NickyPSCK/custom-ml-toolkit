import itertools
from copy import deepcopy
from math import comb
from typing import Generator, List, Optional, Tuple, Any, Union


class CombinatoricFeatureGenerator:
    '''A class to generate combinatoric features from a list of selected columns,
    with optional constraints and budgeting.
    '''
    def __init__(
        self,
        r_start: int,
        r_end: int,
        selected_cols: List[Any],
        required_cols: Optional[List[Any]] = None,
        budget: int = -1,
        verbose: bool = True
    ):
        '''Initialize the generator with given parameters.

        Args:
            r_start (int): Starting size for combinations.
            r_end (int): Ending size for combinations.
            selected_cols (List[Any]): List of columns for generating combinations.
            required_cols (Optional[List[Any]): Required columns for all combinations.
            budget (int): Maximum combinations to generate. Defaults to -1 (no limit).
            verbose (bool): Enable verbose logging. Defaults to True.
        '''
        if r_start <= 0 or r_end <= 0:
            raise ValueError('r_start and r_end must be positive integers.')
        if r_start > r_end:
            raise ValueError('r_start must be less than or equal to r_end.')
        if not selected_cols:
            raise ValueError('selected_cols cannot be empty.')
        if budget < -1:
            raise ValueError('budget must be -1 (unlimited) or a non-negative integer.')

        self._r_start = r_start
        self._r_end = r_end
        self._selected_cols = deepcopy(selected_cols)

        if required_cols is not None:
            self._required_cols = required_cols
        else:
            self._required_cols = list()

        self._budget = budget
        self._verbose = verbose

        self._n = len(self._selected_cols)
        self._melted_required_cols = self.melt_list_of_list(required_cols)
        self._cal_range = self._create_cal_range()
        self._combinations = self._create_combinations()
        self._number_of_combinations = self._cal_number_of_combinations()
        self._remaining = self._number_of_combinations

        self._remaining_budget = budget

    @staticmethod
    def melt_list_of_list(
        list_of_list: Union[List[Any], List[List[Any]]]
    ) -> List[Any]:
        '''Flattens a nested list into a single list.

        Args:
            list_of_list (list): A potentially nested list.

        Returns:
            List: A flattened list.
        '''
        # result_list = list()
        # for member in list_of_list:
        #     if isinstance(member, list):
        #         result_list += CombinatoricFeatureGenerator.melt_list_of_list(member)
        #     else:
        #         result_list.append(member)
        # return result_list

        result_list = list()
        stack = deepcopy(list_of_list)
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            else:
                result_list.append(item)
        return result_list[::-1]

    def _create_cal_range(self) -> range:
        '''Create the range for combination sizes.

        Returns:
            range: A range object for the sizes of combinations.
        '''
        if self._r_start <= self._r_end:
            cal_range = range(self._r_start, self._r_end + 1, 1)
        else:
            cal_range = range(self._r_start, self._r_end - 1, -1)
        return cal_range

    def _cal_number_of_combinations(self) -> int:
        '''Calculate the total number of combinations.

        Returns:
            int: Total number of combinations.
        '''
        no_of_cases = 0
        for r in self._cal_range:
            no_of_cases += comb(self._n, r)
        return no_of_cases

    @property
    def number_of_combinations(self) -> int:
        '''int: Total number of combinations.'''
        return self._number_of_combinations

    @property
    def budget(self) -> int:
        '''int: Current budget for combinations.'''
        return self._budget

    @property
    def n(self) -> int:
        '''int: Number of selected columns.'''
        return self._n

    @property
    def remaining(self) -> int:
        '''int: Number of combinations remaining.'''
        return self._remaining

    @budget.setter
    def budget(self, budget: int):
        '''Set a new budget for combinations.'''
        self._budget = budget
        self._remaining_budget = budget

    def info(self) -> str:
        '''
        Returns:
            str: Information about the current state of the generator.
        '''
        info = [
            '---------------------------------------------------------------',
            f'n: {self.n}'
        ]
        info.extend([f'r = {r}: {comb(self.n, r)} combinations' for r in self._cal_range])
        info.append(f'Remaining {self.remaining}/{self.number_of_combinations}')
        info.append('---------------------------------------------------------------')
        return '\n'.join(info)

    def _create_combinations(self) -> Generator[Tuple, None, None]:
        '''
        Generate combinations of the selected columns.

        Yields:
            Tuple: A combination of selected columns.
        '''
        for r in self._cal_range:
            combinations = itertools.combinations(
                iterable=self._selected_cols,
                r=r
            )
            for combination in combinations:
                yield combination

    def _find_remove_members(self, combination: tuple) -> List[Any]:
        '''Find members removed from the selected columns in a combination.

        Args:
            combination (tuple): The current combination.

        Returns:
            List: Members removed from the selected columns.
        '''
        removed_members = list()
        for member in self._selected_cols:
            if member not in combination:
                removed_members.append(member)
        return removed_members
        # return list(set(self._selected_cols) - set(combination))

    def __next__(self) -> Tuple[list, list]:
        '''Get the next combination.

        Returns:
            Tuple[list, list]: A tuple containing removed members and the combination.

        Raises:
            StopIteration: If no more combinations are available.
        '''
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
            if self._verbose:
                if self._remaining > 0:
                    print(f'Run out of budget, but {self._remaining} combination(s) left.')
                else:
                    print('Run out of combination')
            raise StopIteration()

    def __iter__(self) -> 'CombinatoricFeatureGenerator':
        '''Make the generator iterable.

        Returns:
            CombinatoricFeatureGenerator: The instance itself.
        '''
        return self

    def __len__(self) -> int:
        '''Get the number of remaining combinations.

        Returns:
            int: Number of combinations remaining or within the budget.
        '''
        if self._budget > 0:
            return self._remaining_budget
        else:
            return self._remaining
