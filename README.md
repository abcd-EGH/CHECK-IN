# CHECK-IN: CHECK copyright INfringment

Welcome to **CHECK-IN**, the service dedicated to **CHECK copyright INfringment**. Our mission is to prevent unintentional copyright infringement and raise awareness about copyright issues. Also, by facilitating legal use of copyrighted works, we aim to expand the copyright protection market. This project was carried out in the 3rd term of Korea University's Intelligent Information SW Academy.

This project uses [CRAFT](https://github.com/clovaai/CRAFT-pytorch/blob/master/README.md), an open-source software licensed under the MIT License. For more details on the license, please see the LICENSE file in the project repository or visit [MIT License](https://opensource.org/licenses/MIT).

A demonstration video in Youtube (KOR): https://youtu.be/dMN9y0LOhQQ
A presentation video in Youtube (KOR): https://youtu.be/25_xl5kEbx8
![CHECK-IN_1](https://github.com/abcd-EGH/CHECK-IN/assets/131218154/9a626797-ae6d-4a74-9f83-1d11030ffe9c)
![CHECK-IN_2](https://github.com/abcd-EGH/CHECK-IN/assets/131218154/0af9fcc7-2f70-436b-9162-86f76d6615ff)

## :triangular_flag_on_post:Core Features and Expected Impact

- **Similarity Analysis**: CHECK-IN analyzes the similarity between copyrighted designs (such as movie and performance posters, books, etc.) and user-generated designs. We employ various metrics for this analysis, including cosine similarity, Structural Similarity Index (SSIM), and text area similarity.*CRAFT*, text detector created by **NAVER**, performs text area similarity analysis. However, it is not reflected in the analysis results due to GPU usage issues. This is included by 'APIserver.py'.
- **Prevention of Unintended Copyright Infringement**: Our service is designed to prevent unintentional copyright violations by providing users with a clear analysis of how their designs compare to existing copyrighted works.
- **Enhancement of Copyright Awareness**: By highlighting similarities that might constitute infringement, CHECK-IN educates users about copyright issues, promoting a culture of respect for intellectual property.
- **Promotion of Legal Use**: CHECK-IN encourages the legal use of copyrighted works, contributing to the expansion of the copyright protection market.

## :muscle:What Sets Us Apart

- Unlike other similarity search services, CHECK-IN enhances verification performance not only by measuring cosine similarity but also by evaluating SSIM.
- Also, by measuring the similarity of text areas, we provide a more nuanced and accurate assessment of potential infringement.
- This is particularly important for graphic design images, where plagiarism often involves not just the visual identity of the image but also the positioning of text elements.

### :grey_question:What is 'graphic design images'?

- It refers to an image that conveys information through visual expression and creates an overall design.
- Key features of graphic design image include image type and background, *typography*, design style, and color scheme.

### :rench:difference between MoCov2 model and SimCLR

- [MoCo](https://doi.org/10.48550/arXiv.1911.05722) substituted the Momentum concept into contrast learning. Focusing on the momentum-based update mechanism for the encoder, it enables more consistent and stable feature expression over time. In addition, it has stable performance even at low batch size by utilizing the queue embedded with the negative sample.
- [SimCLR](https://doi.org/10.48550/arXiv.2002.05709) is a model that applies the Large bat size & Longer training, Stronger augmentation, and MLP projection head to SSL and contrast learning.
- [MoCov2](https://doi.org/10.48550/arXiv.2003.04297) is a model that applies the second and third of SimCLR's core to MoCo. In our project, the MoCov2 model extracts feature vectors from images for similarity analysis.

### :mag:Annoy (Approximate Nearest Neighbors Oh Yeah) Algorithm

- [Annoy](https://github.com/spotify/annoy?tab=readme-ov-file), *C++* library with *Python*, is known for its fast search times and minimal memory usage, enabling us to quickly compare user-uploaded designs against a vast database of copyrighted works.
- This algorithm ensures that CHECK-IN can provide quick and accurate result to users, helping to prevent copyright infringement effectively.

### :heavy_plus_sign:Applied technologies in Training

- **Save queues to CPU & Use asynchronously on GPU**: Queue (of MoCov2) is stored on the CPU so that there is no shortage of GPU RAM, and it is brought to the GPU asynchronously only when necessary to minimize performance degradation (mainly speed) caused by data movement.
- **Cosine decay + Warm up**: To improve training speed by reducing the number of epoches required, the training speed is initially gradually increased and then gradually decreased.
- **Early Stopping**: To prevent overfitting, if the number of epoches increases but the train loss does not decrease, then the training will be terminated.
- See 'For_Training_FinalModel_MoCo2.ipynb' in a folder named 'Train&BuildDB' for more detailed training process.

## :dart:For Protecting Creativity

- With CHECK-IN, we are on a mission to protect creativity and promote legal usage of copyrighted material. By understanding and respecting copyright laws, we can all contribute to a richer, more vibrant creative community.

---

For further information, updates, and to start using CHECK-IN, please send an email to 'wlghks7790@gmail.com'.
