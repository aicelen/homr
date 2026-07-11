from homr.model import Staff
from homr.transformer.configs import Config
from training.architecture.transformer.staff2score import Staff2Score
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

inference: Staff2Score | None = None


def parse_staff_tromr(staff_images: NDArray, staffs: Staff, config: Config) -> list[list[EncodedSymbol]]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(config)

    batch_result = inference.predict(staff_images)

    result = []
    for idx, staff in enumerate(staffs):
        if staff.is_grandstaff:
            result.append(batch_result[idx])
        else:
            result.append([r for r in batch_result[idx] if r.position != "lower"])
    return result