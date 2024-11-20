import pandas as pd
import numpy as np
import re
from datetime import datetime

class DataValidator:
    def __init__(self, constraints):
        """
        Initialize the DataValidator with a set of constraints.

        :param constraints: A dictionary where keys are column names and values are dictionaries 
                            specifying constraints (e.g., 'min', 'max', 'allowed_values').
        """
        self.constraints = constraints

    def validate(self, df):
        """
        Validate the DataFrame against the constraints.

        :param df: pandas DataFrame to be validated.
        :return: A dictionary with details of violations for each column.
        """
        violations = {}

        for column, rules in self.constraints.items():
            if column not in df.columns:
                if rules.get("required", False):
                    violations[column] = {"error": "Column not found in DataFrame"}
                continue

            column_violations = []
            column_data = df[column]

            # General Constraints
            if rules.get("not_null", False):
                null_values = column_data[column_data.isnull()]
                if not null_values.empty:
                    column_violations.append({"type": "not_null", "rows": null_values.index.tolist()})

            if rules.get("unique", False):
                duplicated = column_data[column_data.duplicated(keep=False)]
                if not duplicated.empty:
                    column_violations.append({"type": "unique", "rows": duplicated.index.tolist()})

            # Numeric Constraints
            if "min" in rules:
                below_min = column_data[column_data < rules["min"]]
                if not below_min.empty:
                    column_violations.append({"type": "min", "rows": below_min.index.tolist(), "value": rules["min"]})

            if "max" in rules:
                above_max = column_data[column_data > rules["max"]]
                if not above_max.empty:
                    column_violations.append({"type": "max", "rows": above_max.index.tolist(), "value": rules["max"]})

            if rules.get("integer", False):
                non_integers = column_data[~column_data.apply(lambda x: isinstance(x, (int, np.integer)))]
                if not non_integers.empty:
                    column_violations.append({"type": "integer", "rows": non_integers.index.tolist()})

            if rules.get("positive", False):
                non_positive = column_data[column_data <= 0]
                if not non_positive.empty:
                    column_violations.append({"type": "positive", "rows": non_positive.index.tolist()})

            # Categorical Constraints
            if "allowed_values" in rules:
                not_allowed = column_data[~column_data.isin(rules["allowed_values"])]
                if not not_allowed.empty:
                    column_violations.append({
                        "type": "allowed_values", 
                        "rows": not_allowed.index.tolist(),
                        "values": rules["allowed_values"]
                    })

            if "disallowed_values" in rules:
                disallowed = column_data[column_data.isin(rules["disallowed_values"])]
                if not disallowed.empty:
                    column_violations.append({
                        "type": "disallowed_values",
                        "rows": disallowed.index.tolist(),
                        "values": rules["disallowed_values"]
                    })

            # Text Constraints
            if "max_length" in rules:
                too_long = column_data[column_data.astype(str).str.len() > rules["max_length"]]
                if not too_long.empty:
                    column_violations.append({"type": "max_length", "rows": too_long.index.tolist(), "value": rules["max_length"]})

            if "min_length" in rules:
                too_short = column_data[column_data.astype(str).str.len() < rules["min_length"]]
                if not too_short.empty:
                    column_violations.append({"type": "min_length", "rows": too_short.index.tolist(), "value": rules["min_length"]})

            if "pattern" in rules:
                pattern = re.compile(rules["pattern"])
                non_matching = column_data[~column_data.astype(str).str.match(pattern)]
                if not non_matching.empty:
                    column_violations.append({"type": "pattern", "rows": non_matching.index.tolist(), "pattern": rules["pattern"]})

            # Date/Time Constraints
            if "min_date" in rules:
                min_date = pd.to_datetime(rules["min_date"])
                earlier_dates = pd.to_datetime(column_data, errors="coerce") < min_date
                if earlier_dates.any():
                    column_violations.append({"type": "min_date", "rows": earlier_dates[earlier_dates].index.tolist(), "value": str(min_date)})

            if "max_date" in rules:
                max_date = pd.to_datetime(rules["max_date"])
                later_dates = pd.to_datetime(column_data, errors="coerce") > max_date
                if later_dates.any():
                    column_violations.append({"type": "max_date", "rows": later_dates[later_dates].index.tolist(), "value": str(max_date)})

            if "date_format" in rules:
                format_violations = column_data[column_data.apply(lambda x: not self._check_date_format(x, rules["date_format"]))]
                if not format_violations.empty:
                    column_violations.append({"type": "date_format", "rows": format_violations.index.tolist(), "format": rules["date_format"]})

            # Add violations to the result
            if column_violations:
                violations[column] = column_violations

        return violations

    @staticmethod
    def _check_date_format(date_str, date_format):
        """
        Check if a date string matches a specific format.
        """
        try:
            datetime.strptime(date_str, date_format)
            return True
        except (ValueError, TypeError):
            return False

# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        "age": [25, 30, 17, 40],
        "salary": [50000, 60000, 45000, 30000],
        "department": ["HR", "Finance", "IT", "Marketing"],
        "email": ["john.doe@example.com", "jane.doe@", "user@company.com", ""],
        "start_date": ["2023-01-01", "2025-01-01", "invalid", "2022-01-01"]
    }
    df = pd.DataFrame(data)

    # Define constraints
    constraints = {
        "age": {"min": 18, "max": 65, "not_null": True},
        "salary": {"min": 30000, "max": 100000, "positive": True, "integer": True},
        "department": {"allowed_values": ["HR", "Finance", "IT"], "not_null": True},
        "email": {"pattern": r'^[\w\.-]+@[\w\.-]+\.\w+$', "not_null": True},
        "start_date": {"min_date": "2020-01-01", "max_date": "2024-12-31", "date_format": "%Y-%m-%d"}
    }

    # Validate DataFrame
    validator = DataValidator(constraints)
    result = validator.validate(df)

    # Print violations
    if result:
        print("Validation Violations:")
        for col, issues in result.items():
            print(f"Column: {col}")
            for issue in issues:
                print(f"  Type: {issue['type']}, Rows: {issue['rows']}, Constraint: {issue.get('value', issue.get('pattern', issue.get('format')))}")
    else:
        print("No violations found.")
