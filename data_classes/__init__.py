from .constraint_covid import ConstraintCovid
from .covid_mis_20 import CovidMis20
from .constraint_covid_filtered import ConstraintFiltered
from .constraint_covid_appended import ConstraintAppended
from .constraint_covid_filtered_appended import ConstraintFilteredAppended


def dataclass_map(key):
    
    if key == 'aaai-constraint-covid':
        return ConstraintCovid
    if key == 'aaai-constraint-covid-filtered': 
        return ConstraintFiltered
    if key == 'aaai-constraint-covid-appended': 
        return ConstraintAppended
    if key == 'aaai-constraint-covid-filtered-appended': 
        return ConstraintFilteredAppended
    if key == 'covid-misinformation': 
        return CovidMis20