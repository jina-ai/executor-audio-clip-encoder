import os

import torch
import torch.nn.functional as F

from .clip import CLIP
from .clip.clip import tokenize
from .esresnet import ESResNeXtFBSP

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional


ClipFeatures = Tuple[
    Optional[torch.Tensor],  # audio
    Optional[torch.Tensor],  # image
    Optional[torch.Tensor],  # audio
]


ClipLogits = Tuple[
    Optional[torch.Tensor],  # audio x image
    Optional[torch.Tensor],  # audio x text
    Optional[torch.Tensor],  # image x text
]


ClipOutput = Tuple[Tuple[ClipFeatures, ClipLogits], Optional[torch.Tensor]]  # loss


class AudioCLIP(CLIP):
    def __init__(
        self,
        embed_dim: int = 1024,
        # vision
        image_resolution: int = 224,
        vision_layers: Union[Tuple[int, int, int, int], int] = (3, 4, 6, 3),
        vision_width: int = 64,
        vision_patch_size: Optional[int] = None,
        # text
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        # audio
        n_fft: int = 2048,
        hop_length: Optional[int] = 561,
        win_length: Optional[int] = 1654,
        window: Optional[str] = 'blackmanharris',
        normalized: bool = True,
        onesided: bool = True,
        spec_height: int = -1,
        spec_width: int = -1,
        apply_attention: bool = True,
        multilabel: bool = True,
        pretrained: Union[bool, str] = True,
    ):

        super(AudioCLIP, self).__init__(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

        self.audio = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=False,
        )

        self.multilabel = multilabel
        self.pretrained = pretrained

        self.logit_scale_ai = torch.nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_at = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

        if isinstance(self.pretrained, str):
            self.load_state_dict(torch.load(self.pretrained, map_location='cpu'), strict=False)
        elif self.pretrained:
            self.load_state_dict(torch.load(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'CLIP.pt'),
                map_location='cpu'
            ), strict=False)
            print('Image & Text weights loaded')
            try:
                self.audio.load_state_dict(torch.load(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'ESRNXFBSP.pt'),
                    map_location='cpu'
                ), strict=False)
            except RuntimeError as ex:
                print(ex)
                print('Audio weights loaded')

        self.embed_dim = embed_dim

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio(audio.to(self.device))

    def encode_text(
        self,
        text: List[List[str]],
        base_str: str = '{}',
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if batch_indices is not None:
            text = [text[idx] for idx in batch_indices]

        text_joined = [', '.join(entities) for entities in text]
        text_tokens = torch.cat(
            [tokenize(base_str.format(entities)) for entities in text_joined]
        )
        text_tokens = text_tokens.to(self.device)

        return super(AudioCLIP, self).encode_text(text_tokens)

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[List[List[str]]] = None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> ClipOutput:

        audio_features = None
        image_features = None
        text_features = None

        if audio is not None:
            audio_features = self.encode_audio(audio)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        if image is not None:
            image_features = self.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if text is not None:
            if batch_indices is None:
                batch_indices = torch.arange(
                    len(text), dtype=torch.int64, device=self.device
                )

            text_features = self.encode_text(text, '{}', batch_indices)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        features: ClipFeatures = (audio_features, image_features, text_features)

        logit_scale_ai = torch.clamp(self.logit_scale_ai.exp(), min=1.0, max=100.0)
        logit_scale_at = torch.clamp(self.logit_scale_at.exp(), min=1.0, max=100.0)
        logit_scale_it = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)

        logits_audio_image = None
        logits_audio_text = None
        logits_image_text = None

        if (audio_features is not None) and (image_features is not None):
            logits_audio_image = logit_scale_ai * audio_features @ image_features.T

        if (audio_features is not None) and (text_features is not None):
            logits_audio_text = logit_scale_at * audio_features @ text_features.T

        if (image_features is not None) and (text_features is not None):
            logits_image_text = logit_scale_it * image_features @ text_features.T

        logits: ClipLogits = (logits_audio_image, logits_audio_text, logits_image_text)


        return (features, logits), None
