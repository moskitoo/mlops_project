import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from pv_defection_classification.data_drift import reference_df,current_df

from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfMissingValues,
    TestNumberOfDuplicatedRows,
    TestColumnsType,
    TestColumnDrift,
    TestCategoryShare,
    TestCategoryCount
)

# Define a test suite with relevant tests
pv_defect_test_suite = TestSuite(tests=[
    TestNumberOfMissingValues(),
    TestNumberOfDuplicatedRows(),
    TestColumnsType(),
    TestColumnDrift(column_name="x"), 
    TestColumnDrift(column_name="y"),
    TestCategoryShare(column_name="class", category=1, lt=0.3),  # Defect proportion < 30%
    TestCategoryCount(column_name="class", category=1, lt=100)   # Defects detected <100 
])

pv_defect_test_suite.run(reference_data=reference_df, current_data=current_df)
results = pv_defect_test_suite.as_dict()
print(results)
print("All tests passed:", results['summary']['all_passed'])
