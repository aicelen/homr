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
        self.encoder = Encoder(self.config)
        self.decoder = get_decoder(self.config)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError(
                "Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer
            )  # noqa: E501

    def predict(self, image: NDArray, data = None) -> list[EncodedSymbol]:
        """
        Inference an image (NDArray) using Tromr.
        """
        if image is not None:
            print("transforming")
            data = _transform(image)
            np.save("staff.npy", data)
        
        print(data.shape)
        # data = np.flip(data, axis=0)

        # Create special tokens
        start_token  = np.ones((BATCH_SIZE, 1), dtype=np.int64)
        nonote_token = np.zeros((BATCH_SIZE, 1), dtype=np.int64)

        t0 = perf_counter()
        # Generate context with encoder. The encoder and decoder may run in
        # different precisions (e.g. the CoreML encoder is fp16 while the
        # decoder stays on the fp32 CPU model), so cast the context to the
        # dtype the decoder expects before handing it over.
        context = self.encoder.generate(data)
        context_dtype = np.float16 if self.decoder.fp16 else np.float32
        if context.dtype != context_dtype:
            context = context.astype(context_dtype)

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
        arr = arr[:, np.newaxis, :, :]
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

def test_transformer_on_npy(path_to_npy: str) -> None:
    """
    Tests the transformer on an image and prints the results.
    Args:
        path_to_img(str): Path to the image to test
    """

    model = Staff2Score(Config())
    data = np.load(path_to_npy)
    print(data.shape)
    out = model.predict(None, data)
    eprint(out)


if __name__ == "__main__":
    import sys

    test_transformer_on_npy(sys.argv[1])
