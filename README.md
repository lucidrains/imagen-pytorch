<img src="./imagen.png" width="450px"></img>

## Imagen - Pytorch (wip)

Implementation of <a href="https://gweb-research-imagen.appspot.com/">Imagen</a>, Google's Text-to-Image Neural Network that beats DALL-E2, in Pytorch, the new SOTA for text-to-image.

Architecturally, it is actually much simpler than DALL-E2. It composes of a cascading DDPM conditioned on text embeddings from a large pretrained T5 model (attention network). It also contains improvements to the clipping for improved classifier free guidance, as well as noise level conditioning.

It appears neither CLIP nor prior network is needed after all. And so research continues.

## Citations

```bibtex
@misc{Saharia2022,
    title   = {Imagen: unprecedented photorealism × deep level of language understanding}, 
    author  = {Chitwan Saharia*, William Chan*, Saurabh Saxena†, Lala Li†, Jay Whang†, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho†, David Fleet†, Mohammad Norouzi*},
    year    = {2022}
}
```
