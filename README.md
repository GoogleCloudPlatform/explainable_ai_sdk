# Explainable AI SDK

This is a Python SDK for
[Google Cloud Explainable AI](https://cloud.google.com/explainable-ai), an
explanation service that provides insight into machine learning models deployed
on [AI Platform](https://cloud.google.com/ai-platform). The Explainable AI SDK
helps to visualize explanation results, and to define _explanation metadata_ for
the explanation service.

Explanation metadata tells the explanation service which of your model's inputs
and outputs to use for your explanation request. The SDK has metadata builders
that help you to build and save an explanation metadata file before you deploy
your model to AI Platform.

The Explainable AI SDK also helps you to visualize feature attribution results
on models deployed to AI Platform.

## Installation

The Explainable AI SDK supports models built with:

- Python 3.7 and later
- TensorFlow 1.15 or TensorFlow 2.x.

The Explainable AI SDK is preinstalled on
[Google Cloud AI Platform Notebooks](https://cloud.google.com/ai-platform-notebooks)
images.

For other platforms:

1. Make sure that you have
   [installed Cloud SDK](https://cloud.google.com/sdk/docs/quickstarts). In
   order to communicate with Cloud AI Platform, the Explainable AI SDK requires
   a shell environment with Cloud SDK installed.

1. Install the Explainable AI SDK:

    ```shell
    pip install explainable-ai-sdk
    ```

## Metadata Builders

After you build your model, you use a metadata builder to create your
explanation metadata. This produces a JSON file that is used for model
deployment on AI Platform.

There are different metadata builders for TensorFlow 1.x and 2.x in
their respective folders.

### TensorFlow 2.x

For TensorFlow 2.x, there is one metadata builder that takes a
**SavedModel**, and uploads both your model and metadata file to Cloud Storage.

For example:

```python
from explainable_ai_sdk.metadata.tf.v2 import SavedModelMetadataBuilder
builder = SavedModelMetadataBuilder(
    model_path)
builder.save_model_with_metadata('gs://my_bucket/model')  # Save the model and the metadata.
```

### TensorFlow 1.x

For TensorFlow 1.x, the Explainable AI SDK supports models built with Keras,
Estimator and the low-level TensorFlow API. There is a different metadata
builder for each of these three TensorFlow APIs. An example usage for a Keras
model would be as follows:

```python
from explainable_ai_sdk.metadata.tf.v1 import KerasGraphMetadataBuilder
my_model = keras.models.Sequential()
my_model.add(keras.layers.Dense(32, activation='relu', input_dim=10))
my_model.add(keras.layers.Dense(32, activation='relu'))
my_model.add(keras.layers.Dense(1, activation='sigmoid'))
builder = KerasGraphMetadataBuilder(my_model)
builder.save_model_with_metadata('gs://my_bucket/model')  # Save the model and the metadata.
```

For examples using the Estimator and TensorFlow Core builders, refer to the
[v1 README file](./explainable_ai_sdk/metadata/tf/v1/README.md).

### Making Predict and Explain Calls

The Explainable AI SDK includes a model interface to help you communicate with
the deployed model more easily. With this interface, you can call `predict()`
and `explain()` functions to get predictions and explanations for the provided
data points, respectively.

Here is an example snippet for using the model interface:

```python
project_id = "example_project"
model_name = "example_model"
version_name = "v1"

m = explainable_ai_sdk.load_model_from_ai_platform(project_id, model_name, version_name)
instances = []

# ... steps for preparing instances

predictions = m.predict(instances)
explanations = m.explain(instances)
```

### Explanation, Attribution, and Visualization

The `explain()` function returns a list of `Explanation` objects --
one `Explanation` per input instance. This object makes it easier to interact
with returned attributions. You can use the `Explanation` object to get
feature importance and raw attributions, and to visualize attributions.

**Note**: Currently, the `feature_importance()` and `as_tensors()` functions
only work on tabular models, due to the limited payload size. We are working on
making both functions available for image models.**

#### Get feature importance

The `feature_importance()` function returns the imporance of each feature
based on feature attributions. Note that if a feature has more than one
dimension, the importance is calculated based on the aggregation.

```python
explanations[0].feature_importance()
```

#### Get raw attributions

To get feature attributions over each dimension, use the `as_tensors()`
function to return the raw attributions as tensors.

```python
explanations[0].as_tensors()
```

#### Visualize attributions

The `Explanation` class allows you to visualize feature attributions directly.
For both image and tabular models, you can call `visualize_attributions()`
to see feature attributions.

```python
explantions[0].visualize_attributions()
```

Here is an example visualization:

![An attribution visualization for a tabular model](http://services.google.com/fh/files/misc/explainable_ai_sdk_tabular_attributions_visualzation.png)


## Caveats

* This library works with (and depends) on either major version of TensorFlow.
* Do not import the `metadata/tf/v1` and `metadata/tf/v2` folders in the
  same Python runtime. If you do, there may be unintended side effects of mixing
  TensorFlow 1.x and 2.x behavior.

## Explainable AI documentation

For more information about Explainable AI, refer to the
[Explainable AI documentation](https://cloud.google.com/ai-platform/prediction/docs/ai-explanations/overview).

## License

All files in this repository are under the
[Apache License, Version 2.0](https://github.com/GoogleCloudPlatform/explainable_ai_sdk/blob/master/LICENSE)
unless noted otherwise.

**Note:** We are not accepting contributions at this time.
