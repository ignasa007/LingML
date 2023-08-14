from .constraint_covid import ConstraintCovid
from .covid_mis_20 import CovidMis20
from .constraint_covid_cleaned import Constraintcleaned
from .constraint_covid_appended import ConstraintAppended
from .constraint_covid_cleaned_appended import ConstraintcleanedAppended


def dataclass_map(key):
    
    if key == 'aaai-constraint-covid':
        return ConstraintCovid
    if key == 'aaai-constraint-covid-cleaned': 
        return Constraintcleaned
    if key == 'aaai-constraint-covid-appended': 
        return ConstraintAppended
    if key == 'aaai-constraint-covid-cleaned-appended': 
        return ConstraintcleanedAppended
    if key == 'covid-misinformation': 
        return CovidMis20