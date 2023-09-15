# Crack Surface Classification
This project focuses on Crack Surface Classification and it employs the TensorFlow framework. This project also uses a transfer learning model, which is MobileNetV2.
## Context
&emsp Crack detection plays a pivotal role in the field of structural building for several compelling reasons. These cracks, often subtle at their inception, can serve as early indicators of underlying structural problems. Addressing them promptly is vital to prevent more extensive damage, ensure the safety of occupants, and ultimately, save costs on extensive repairs or even potential structural failures. In addition to immediate safety concerns, efficient crack detection also contributes to the long-term durability of buildings and infrastructure, promoting their sustainability. It enables engineers and maintenance teams to adopt proactive measures, extending the lifespan of structures and enhancing their overall performance. As such, crack detection is an essential component of modern construction and maintenance practices, ensuring the safety, integrity, and longevity of our built environment.
## Dataset
The dataset comprises images capturing diverse concrete surfaces, some exhibiting cracks and others devoid of any. These images are categorized into two distinct folders for the purpose of image classification: one folder contains negative instances (those without cracks), while the other folder contains positive instances (those with cracks). Each class comprises a set of 20,000 images, resulting in a total dataset size of 40,000 images, all with dimensions of 227 x 227 pixels and RGB channels. The dataset originates from 458 high-resolution images, each boasting a resolution of 4032x3024 pixels, and it was generated using the methodology proposed by Zhang et al. in 2016. It is worth noting that the high-resolution images exhibit significant variability in terms of surface finish and illumination conditions.
<br>This dataset is taken from the website Mendeley Data - [Crack Detection, contributed by Çağlar Fırat Özgenel.](https://gaganpreetkaurkalsi.netlify.app/)
> Reference
> <br> Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, V2, doi: 10.17632/5y9wdsg2zt.2

<br> ![combination](https://github.com/Fahmie23/crack_surface_classification/assets/130896959/f0718b15-c215-4fd1-adf9-83a3c6ac3b7d)

## Architecture of the Model
The architecture of the model when using the transfer learning model, which is MobileNetV2.
<br> ![model](https://github.com/Fahmie23/crack_surface_classification/assets/130896959/9a1df142-0b82-4e93-a86d-784ac6540f06)

## Performance of the Model
Before training the model and just using the transfer learning model, this is the model evaluation.
![before_train](https://github.com/Fahmie23/crack_surface_classification/assets/130896959/d74057c4-b804-4002-8f40-cb93dd9febf3)

<br>After training the model, this is the model evaluation.
![after_train](https://github.com/Fahmie23/crack_surface_classification/assets/130896959/3ec2273d-eae7-48a8-ae1f-c2c6727b9ee0)

<br>The graph plot for loss and validation loss.
![loss](https://github.com/Fahmie23/crack_surface_classification/assets/130896959/e02ba81e-5ef0-4a72-9418-d7e61fed5a55)

<br>The graph plot for accuracy and validation accuracy.
![accuracy](https://github.com/Fahmie23/crack_surface_classification/assets/130896959/b64d6a3b-2776-458c-a0ad-570a31d548e9)

## Results
![output](https://github.com/Fahmie23/crack_surface_classification/assets/130896959/56d15752-49a0-4ea9-a088-c600b9838864)

