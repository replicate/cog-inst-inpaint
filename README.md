# Cog-Inst-Inpaint

This is an implementation of [Inst-Inpaint](https://github.com/abyildirim/inst-inpaint/) as a [Cog](https://github.com/replicate/cog) model. Inst-Inpaint is a diffusion based model that performs text-guided object removal from images. For more details, see this [Replicate model](https://replicate.com/alaradirik/inst-inpaint), [paper](https://arxiv.org/abs/2304.03246) and [project website](https://instinpaint.abyildirim.com/).

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your fork or other models to [Replicate](https://replicate.com).

## Basic Usage

To run a prediction:

```bash
cog predict -i image=@cups.webp -i instruction="remove the cup on the left"
```

```bash
cog run -p 5000 python -m cog.server.http
```

## References
```
@misc{yildirim2023instinpaint,
      title={Inst-Inpaint: Instructing to Remove Objects with Diffusion Models}, 
      author={Ahmet Burak Yildirim and Vedat Baday and Erkut Erdem and Aykut Erdem and Aysegul Dundar},
      year={2023},
      eprint={2304.03246},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```