__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import librosa
import numpy as np

from jina import Flow, Document, DocumentArray
from audio_clip_encoder import AudioCLIPEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_flow_from_yml():

    doc = DocumentArray([Document()])
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        resp = f.post(on='test', inputs=doc, return_results=True)

    assert resp is not None


def test_embedding_exists():

    x_audio, _ = librosa.load(os.path.join(cur_dir, '../data/sample.mp3'))
    doc = DocumentArray([Document(blob=x_audio)])

    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        responses = f.post(on='index', inputs=doc, return_results=True)

    assert responses[0].docs[0].embedding is not None and responses[0].docs[0].embedding.shape == (1024, )


def test_many_documents():

    audio1, _ = librosa.load(os.path.join(cur_dir, '../data/sample.mp3'))
    audio2, _ = librosa.load(os.path.join(cur_dir, '../data/sample.wav'))
    doc = DocumentArray([Document(blob=audio1), Document(blob=audio2)])

    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        responses = f.post(on='index', inputs=doc, return_results=True)

    assert responses[0].docs[0].embedding.shape == (1024, ) and responses[0].docs[1].embedding.shape == (1024, )

def test_traversal_paths():

    audio1, _ = librosa.load(os.path.join(cur_dir, '../data/sample.mp3'))
    audio2, _ = librosa.load(os.path.join(cur_dir, '../data/sample.wav'))

    audio1_chunks = np.split(audio1, 4)
    audio2_chunks = np.split(audio2, 2)

    docs = [
        Document(
            id='root1',
            blob=audio1,
            chunks=[
                Document(id=f'chunk1{i}', blob=chunk) for i, chunk in enumerate(audio1_chunks)
            ]
        ),
        Document(
            id='root2',
            blob=audio2,
            chunks=[
                Document(id='chunk21', blob=audio2_chunks[0]),
                Document(id='chunk22', blob=audio2_chunks[1], chunks=[
                    Document(id=f'chunk22{i}', blob=chunk) for i, chunk in enumerate(np.split(audio2_chunks[1], 3))
                ])
            ]
        )
    ]
    f = Flow().add(uses={
        'jtype': AudioCLIPEncoder.__name__,
        'with': {
            'default_traversal_paths': ['c'],
        }
    })
    with f:
        results = f.post(on='/test', inputs=docs, return_results=True)
        for path, count in [['r', 0], ['c', 6], ['cc', 0]]:
            embeddings = DocumentArray(results[0].docs).traverse_flat([path]).get_attributes('embedding')
            assert all(
                embedding.shape == (1024,)
                for embedding in embeddings
            ) and len(embeddings) == count

        results = f.post(on='/test', inputs=docs, parameters={'traversal_paths': ['cc']}, return_results=True)
        for path, count in [['r', 0], ['c', 0], ['cc', 3]]:
            embeddings = DocumentArray(results[0].docs).traverse_flat([path]).get_attributes('embedding')
            assert all(
                embedding.shape == (1024,)
                for embedding in embeddings
            ) and len(embeddings) == count
