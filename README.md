![WHY2](https://user-images.githubusercontent.com/25187211/195958295-71e9d8a9-35c0-4491-ad6d-5e277b0c3ce8.png)

### Framework-agnostic XAI Library for Computer Vision, for understanding why models behave that way.



#### Gradcam Methods

- GradCam   : Implemented for classification problems, for PyTorch and Tensorflow/Keras.
- GradCam++ : Implementation on the way.

#### Weights Analysis

- To be done

#### How to run? 

```

from why import Explain
from PIL import Image

filename = "my_perfect_image.png"

original_image = Image.open(filename)
preprocessed_image = preprocess_image(original_image)

raw_model = keras.applications.EfficientNetV2B0(weights='imagenet', include_top=True)

why_explain = Explain(raw_model)
heatmap = why_explain.explain(preprocessed_image,explain_class=999)

overlay_heatmap = why_explain.overlay_heatmap(original_image, heatmap, filename="my_saving_path.png", image_size=(1024,1024), alpha=0.5, colormap_name="jet", return_bytes=False)


```
![whytf](https://user-images.githubusercontent.com/25187211/195960081-200dea25-3522-4ece-917f-31ab6cc8196a.png)



```
extract_xai_area = why_explain.extract_area(
        preprocessed_image,
        original_image,
        threshold=0.85,
        explain_class=999,
        method="GradCam",
    )


```

![extract](https://user-images.githubusercontent.com/25187211/195960247-43c849a0-d2a2-4a1b-9091-c5f87191c9f7.png)


```
segment_xai_area_coordinates = why_explain.annotate(
        img_batch,
        imgorig.size,
        threshold=0.85,
        explain_class=None,
        method="GradCam",
    )
        
im = plt.imread(filename)
implot = plt.imshow(im)
for p,q in  [(x["x"],x["y"]) for x in segment_xai_area_coordinates]:
    x_cord = p # try this change (p and q are already the coordinates)
    y_cord = q
    plt.scatter([x_cord], [y_cord])

plt.savefig("my_segment_area.png)
plt.clf() 

```

![segment](https://user-images.githubusercontent.com/25187211/195960258-0a36f36c-83e6-4e47-8c9b-61621392b42d.png)


