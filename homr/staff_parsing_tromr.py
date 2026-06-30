from homr.model import Staff
from homr.transformer.configs import Config
from homr.transformer.staff2score import Staff2Score
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

inference: Staff2Score | None = None


def parse_staff_tromr(staff_image: NDArray, staff: Staff, config: Config) -> list[EncodedSymbol]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(config)

    result = inference.predict(staff_image)
    if staff.is_grandstaff:
        return result
    return [r for r in result if r.position != "lower"]
