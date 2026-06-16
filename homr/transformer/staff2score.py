import os
from time import perf_counter

import numpy as np
from PIL import Image

from homr.simple_logging import eprint
from homr.transformer.configs import Config, BATCH_SIZE
from homr.transformer.decoder_inference import get_decoder
from homr.transformer.encoder_inference import Encoder
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


class Staff2Score:
    """
    Inference class for Tromr. Use predict() for prediction
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.use_gpu_inference = False
        self.encoder = Encoder(self.config)
        self.decoder = get_decoder(self.config)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError(
                "Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer
            )  # noqa: E501

    def predict(self, image: NDArray) -> list[EncodedSymbol]:
        """
        Inference an image (NDArray) using Tromr.
        """
        data = np.tile(_transform(image=image), (BATCH_SIZE, 1, 1, 1))
        print(data.shape)

        # Create special tokens
        start_token  = np.ones((BATCH_SIZE, 1), dtype=np.int64)
        nonote_token = np.zeros((BATCH_SIZE, 1), dtype=np.int64)
        
        # Make a prediction using decoder
        self.decoder.generate(
            start_token,
            nonote_token,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=self.encoder.generate(data),
        )

        t0 = perf_counter()
        # Generate context with encoder
        context = self.encoder.generate(data)

        # Make a prediction using decoder
        out = self.decoder.generate(
            start_token,
            nonote_token,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
        )

        eprint(f"Inference Time Tromr: {perf_counter()-t0}")

        return out


class ConvertToArray:
    def __init__(self) -> None:
        self.mean = np.array([0.7931]).reshape(1, 1, 1)
        self.std = np.array([0.1738]).reshape(1, 1, 1)

    def normalize(self, array: NDArray) -> NDArray:
        return (array - self.mean) / self.std

    def __call__(self, image: NDArray) -> NDArray:
        arr = np.array(image) / 255
        arr = arr[np.newaxis, np.newaxis, :, :]
        return self.normalize(arr).astype(np.float32)


_transform = ConvertToArray()


def test_transformer_on_image(path_to_img: str) -> None:
    """
    Tests the transformer on an image and prints the results.
    Args:
        path_to_img(str): Path to the image to test
    """

    model = Staff2Score(Config())
    image = Image.open(path_to_img)
    out = model.predict(np.array(image))
    eprint(out)


if __name__ == "__main__":
    import sys

    test_transformer_on_image(sys.argv[1])
