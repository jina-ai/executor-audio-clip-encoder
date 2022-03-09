import copy

from audio_clip.model import AudioCLIP
from audio_clip_v2.model import AudioCLIP as AudioCLIPv2
import torch

import pytest
import os
import librosa
from audio_clip_v2.model.esresnet.fbsp import _ESResNetFBSP

cur_dir = os.path.dirname(os.path.abspath(__file__))
cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def old_model():
    return AudioCLIP(
        pretrained=os.path.join(cur_dir, '../../.cache/AudioCLIP-Full-Training.pt')
    ).to('cpu')


@pytest.fixture()
def new_model():
    return (
        AudioCLIPv2(
            pretrained=os.path.join(cur_dir, '../../.cache/AudioCLIP-Full-Training.pt')
        )
        .to('cpu')
        .eval()
    )


@pytest.fixture()
def audio_input():
    x_audio, _ = librosa.load(os.path.join(cur_dir, '../test_data/sample.mp3'))
    return x_audio


@pytest.fixture()
def text_input():
    return ['Hello', 'beautiful!']


def test_same_output_audio_0(new_model, audio_input):
    old_model = AudioCLIPv2(
        pretrained=os.path.join(cur_dir, '../../.cache/AudioCLIP-Full-Training.pt')
    ).to('cpu')

    with torch.inference_mode():
        x_audio = torch.tensor([audio_input, audio_input])
        embedding_v2 = new_model.encode_audio(audio=x_audio)[0]
        embedding_v1 = old_model.encode_audio(audio=x_audio)[0]
        assert torch.allclose(embedding_v1, embedding_v2)


@torch.inference_mode()
def test_same_output_audio(old_model, new_model, audio_input):
    x_audio = torch.tensor([audio_input, audio_input])
    embedding_v1 = old_model.encode_audio(audio=x_audio)[0]
    embedding_v2 = new_model.encode_audio(audio=x_audio)[0]
    assert torch.allclose(embedding_v1, embedding_v2)


@torch.inference_mode()
def test_batchsize_one_audio(new_model, audio_input):
    x_audio_double = torch.tensor([audio_input, audio_input])
    x_audio_single = torch.tensor([audio_input])
    embedding_single = new_model.encode_audio(audio=x_audio_single)[0]
    embedding_double = new_model.encode_audio(audio=x_audio_double)[0]
    assert torch.allclose(embedding_single[0], embedding_double[0])


def test_same_weights(old_model, new_model, audio_input):

    with torch.no_grad():
        for param1, param2 in zip(new_model.parameters(), old_model.parameters()):
            assert param1.data.ne(param2.data).sum() == 0



@torch.inference_mode()
def test_same_output_text(old_model, new_model, text_input):
    embedding_v1 = old_model.encode_text(text_input)[0]
    embedding_v2 = new_model.encode_text(text_input)[0]
    assert torch.equal(embedding_v1, embedding_v2)


@torch.inference_mode()
def test_same_output_text(old_model, new_model, text_input):
    model = _ESResNetFBSP()
    model_eval = _ESResNetFBSP().eval()

    assert torch.equal(embedding_v1, embedding_v2)
