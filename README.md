
# Attention Guided Textual Inversion Based Personalized Generation
This repo combines the work done in two papers:
## An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2208.01618)

[[Project Website](https://textual-inversion.github.io/)]

> **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**<br>
> Rinon Gal<sup>1,2</sup>, Yuval Alaluf<sup>1</sup>, Yuval Atzmon<sup>2</sup>, Or Patashnik<sup>1</sup>, Amit H. Bermano<sup>1</sup>, Gal Chechik<sup>2</sup>, Daniel Cohen-Or<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>NVIDIA

This work uses textual inversion to perform personalized generation, as can be seen in the [official repo]("https://github.com/rinongal/textual_inversion"), from which this repo is forked.

>**Abstract**: <br>
> Text-to-image models offer unprecedented freedom to guide creation through natural language.
  Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes.
  In other words, we ask: how can we use language-guided models to turn <i>our</i> cat into a painting, or imagine a new product based on <i>our</i> favorite toy?
  Here we present a simple approach that allows such creative freedom.
  Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model.
  These "words" can be composed into natural language sentences, guiding <i>personalized</i> creation in an intuitive way.
  Notably, we find evidence that a <i>single</i> word embedding is sufficient for capturing unique and varied concepts.
  We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks.


## Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models
[![arXiv]](https://arxiv.org/abs/2301.13826)

Hila Chefer∗ Yuval Alaluf∗ Yael Vinker Lior Wolf Daniel Cohen-Or
> **Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models**<br>
> Hila Chefer<sup>1,2</sup>, Yuval Alaluf<sup>1</sup>, Yael Vinker<sup>2</sup>, Lior Wolf<sup>1</sup>, Daniel Cohen-Or<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University

This work uses tackles the problem of catastrophic neglect (missing objects or wrongfully attributed traits in generation) by guiding the generation process based on the attention maps at inference time. The approach is to encourage a state where at each denoising step, for at least some components of the output sequence, the lowest weighted input token (as determined by the attention layers) receives the highest weight possible. The code for this work is accessible at the [official repo]("[https://github.com/rinongal/textual_inversion](https://github.com/AttendAndExcite/Attend-and-Excite)").

>**Abstract**: <br>
> Recent text-to-image generative models have demonstrated an unparalleled ability to generate diverse and creative imagery guided by a target text prompt. While revolutionary, current state-of-the-art diffusion models may still
fail in generating images that fully convey the semantics in
the given text prompt. We analyze the publicly available
Stable Diffusion model and assess the existence of catastrophic neglect, where the model fails to generate one or
more of the subjects from the input prompt. Moreover, we
find that in some cases the model also fails to correctly
bind attributes (e.g., colors) to their corresponding subjects. To help mitigate these failure cases, we introduce the
concept of Generative Semantic Nursing (GSN), where we
seek to intervene in the generative process on the fly during inference time to improve the faithfulness of the generated images. Using an attention-based formulation of GSN,
dubbed Attend-and-Excite, we guide the model to refine the cross-attention units to attend to all subject tokens in the text prompt and strengthen — or excite — their activations,
encouraging the model to generate all subjects described
in the text prompt. We compare our approach to alternative approaches and demonstrate that it conveys the desired
concepts more faithfully across a range of text prompts.

# Mixing The Two
The Attend and Excite code is primarily designed for guiding the generation process for latent diffusion models from the diffusers module. The latent diffusion model used in "An Image Is Worth One Word" is structured differently (class attributes, names architecture and code structure) and therefore requires adaptations not only in code, but also in hyperparameters applied in both the sampling process and the attend and excite pipeline implementation.
For comparison, the main changes in this repo (in regards to the original Textual Inversion repo) are within the implemenation of the DDIM and DDMP samplers, attention.py implemenatations of attention layers, openaimodel.py diffusion models implementations and withing the BERT text encoder (ldm/modules/encoders/modules.py).

