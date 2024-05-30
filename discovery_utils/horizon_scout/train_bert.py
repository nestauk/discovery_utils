from discovery_utils.getters.horizon_scout import get_training_data
from discovery_utils.horizon_scout.utils import get_current_datetime


if __name__ == "__main__":
    print(get_current_datetime())  # noqa: T001
    asf = get_training_data("ASF")
    print(asf)  # noqa: T001
