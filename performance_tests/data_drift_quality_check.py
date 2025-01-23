import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from pv_defection_classification.data_drift import reference_df,current_df
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues

# Create a test suite to check for missing values
data_test = TestSuite(tests=[TestNumberOfMissingValues()])

# Run tests on reference and current data
data_test.run(reference_data=reference_df, current_data=current_df)

# Get the results in dictionary format
result = data_test.as_dict()

# Print the test results
print(result)
print("All tests passed: ", result['summary']['all_passed'])

