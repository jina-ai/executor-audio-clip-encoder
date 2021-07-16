__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import librosa

from jina import Executor, Document, DocumentArray

from audio_clip_encoder import AudioCLIPEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_load():
    encoder = Executor.load_config(os.path.join(cur_dir, '../../config.yml'))
    assert encoder.model_path.endswith('AudioCLIP-Full-Training.pt')


def test_embedding_dimension():
    x_audio, sample_rate = librosa.load(os.path.join(cur_dir, '../data/sample.wav'))
    doc = DocumentArray([Document(blob=x_audio)])
    model = AudioCLIPEncoder()
    model.encode(doc, parameters={})
    assert doc[0].embedding.shape == (1024, )
